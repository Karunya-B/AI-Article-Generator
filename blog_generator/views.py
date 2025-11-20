from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from dotenv import load_dotenv
load_dotenv()
import json,re
import logging
import requests
from django.conf import settings
from youtube_transcript_api import YouTubeTranscriptApi

from pytubefix import YouTube
import os
import assemblyai as aai
import google.generativeai as genai
from typing import Optional
from .models import BlogPost
@login_required
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def generate_blog(request):
    if request.method == 'POST':
        try:
           data = json.loads(request.body)
           yt_link = data['link']
          
        except(KeyError, json.JSONDecodeError):
           return JsonResponse({'error': 'Invalid data'}, status=400)
        
        title = yt_title(yt_link)
        
        #get transcript
        transcription = get_transcription(yt_link)
        if not transcription:
            return JsonResponse({'error': "failed to get transcript"}, status=500)
        #generate blog using genai
        blog_content = generate_blog_content(transcription)
        if not blog_content:
            return JsonResponse({'error': "failed to generate blog content"}, status=500)
        
        #save blog to db
        new_blog_article = BlogPost.objects.create(
            user=request.user,
            youtube_title=title,
            youtube_link=yt_link,
            generated_content=blog_content
        )
        new_blog_article.save()
        #return blog content as json response
        return JsonResponse({'title': title, 'content': blog_content})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
# Get YouTube title (using transcript API fallback)
def extract_id(url):
    pattern = r"(?:v=|be/|embed/)([\w-]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None
def yt_title(link):
    try:
        yt = YouTube(link)
        return yt.title
    except Exception:
        return "Unknown Title"
def download_audio(link):
    yt = YouTube(link)
    audio = yt.streams.filter(only_audio=True).first()
    out_file = audio.download(output_path=settings.MEDIA_ROOT)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.replace(out_file, new_file)
    return new_file
def get_transcription(link):
    audio_file=download_audio(link)
    aai.settings.api_key = os.getenv("AAI_API_KEY")

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    return transcript.text
logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def _safe_extract_text_from_response(resp) -> Optional[str]:
    """Try several common shapes to extract text from SDK/REST responses."""
    # 1) try convenient accessor (may raise ValueError)
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except ValueError:
        logger.debug("response.text accessor raised ValueError; falling back to manual parsing")

    # 2) try SDK-style candidates attribute
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates:
            # iterate candidates, try common nested shapes
            for c in candidates:
                content = getattr(c, "content", None)
                if content:
                    # content often is a list of parts or dicts
                    try:
                        first = content[0]
                        # dict-like shapes
                        if isinstance(first, dict) and "text" in first:
                            return first["text"].strip()
                        if isinstance(first, dict) and "content" in first:
                            inner = first["content"][0]
                            if isinstance(inner, dict) and "text" in inner:
                                return inner["text"].strip()
                        # object-like shapes — attempt attribute access
                        if hasattr(first, "get") and first.get("text"):
                            return first.get("text").strip()
                    except Exception:
                        continue
    except Exception as e:
        logger.debug("Error parsing candidates attribute: %s", e)

    # 3) dict-like fallback (for REST responses)
    try:
        j = dict(resp)
    except Exception:
        j = None
    if j:
        try:
            return j["candidates"][0]["content"][0]["text"].strip()
        except Exception:
            pass

    return None


def _rest_generate(prompt: str, model_name: str = "gemini-2.5-flash") -> Optional[str]:
    """
    REST fallback for newer model names. Returns text or None.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "temperature": 0.2,
        "maxOutputTokens": 1600
    }
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    j = r.json()
    try:
        return j["candidates"][0]["content"][0]["text"].strip()
    except Exception as e:
        logger.warning("REST generate returned unexpected shape: %s", e)
        return None

def generate_blog_content(transcription:str)-> str:
    prompt = (
        "You are an expert content writer. Based on the transcript below, write a long, "
        "well-structured blog article with title, intro, several subheadings, and a conclusion. "
        "Do NOT mention 'YouTube' or 'transcript and any other text other than blog article'.\n\n"
        f"{transcription}\n\nArticle:"
    )

    sdk_model_name = "models/gemini-2.5-pro"
    try:
        model = genai.GenerativeModel(sdk_model_name)
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 1200, "temperature": 0.2})
        text = _safe_extract_text_from_response(response)
        if text:
            return text

        # retry with more tokens
        logger.info("No text found in first SDK response; retrying with larger token budget.")
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 2400, "temperature": 0.2})
        text = _safe_extract_text_from_response(response)
        if text:
            return text

        logger.warning("SDK responses contained no usable text; trying REST fallback.")
    except Exception as e:
        logger.exception("SDK generation failed; attempting REST fallback: %s", e)

    # REST fallback
    try:
        rest_text = _rest_generate(prompt, model_name="gemini-2.5-flash")
        if rest_text:
            return rest_text
    except Exception as e:
        logger.exception("REST fallback failed: %s", e)

    return "Sorry — generation failed. Please try again."
def blog_list(request):
    blog_articles = BlogPost.objects.filter(user=request.user).order_by('-id')
    return render(request, 'all-blogs.html', {'blog_articles': blog_articles})
def blog_details(request, pk):
    blog_article_detail = BlogPost.objects.get(id=pk)
    if request.user == blog_article_detail.user:
        return render(request, 'blog-details.html', {'blog_article_detail': blog_article_detail})
    else:
        return redirect('/')
def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('/')
        else:
            return render(request, 'login.html', {'error_message': 'Invalid credentials'})
    return render(request, 'login.html')
def user_signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirmpassword = request.POST['confirmpassword']

        if password == confirmpassword:
            try:
                user = User.objects.create_user(username, email, password)
                user.save()
                login(request, user)
                return redirect('/')
               

            except Exception as e:
                  return render(request, 'signup.html', {'error_message': str(e)})

        else:
            return render(request, 'signup.html', {'error_message': 'Passwords do not match'})

    return render(request, 'signup.html')   
def user_logout(request):
    logout(request)
    return redirect('/')

