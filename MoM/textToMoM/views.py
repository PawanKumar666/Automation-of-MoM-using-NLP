from django.shortcuts import render
from django.http import HttpResponse

from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch
import matplotlib.pyplot as plt
import re
# Create your views here.
import os
#from google.cloud.speech import SpeechClient, RecognitionAudio, RecognitionConfig

def index(request):
    return render(request,"index.html")

def get_summarization(data):
  data = data.split('\n')
  for entry in data:
    if len(entry) == 0:
      data.remove(entry)
  input_text = process_data(data,name_required = False)
  summarized_text = get_summary(input_text)
  return summarized_text

def process_data(data,name_required = False):
  res = []
  if not name_required:
    for entry in data:
      pos = entry.find(':')
      res.append(entry[pos+1:])
      data = res
  return '.'.join(data)

def get_summary(input_data):

  tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
  input_ids = tokenizer(input_data, return_tensors="pt").input_ids.to("cuda")
  global_attention_mask = torch.zeros_like(input_ids)
  global_attention_mask[:, 0] = 1
  model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv", return_dict_in_generate=True).to("cuda")
  sequences = model.generate(input_ids, global_attention_mask=global_attention_mask).sequences
  summary = tokenizer.batch_decode(sequences)
  mod_summary = summary[0].split('#')[0]
  mod_summary = mod_summary.replace('</s>','')
  mod_summ_list = mod_summary.split('.')
  mod_summ_upd = list(set(mod_summ_list))
  mod_result = '.'.join(mod_summ_upd)
  mod_result[0].upper()
  return mod_result

def process_text(request):
    data = request.POST['text']
    # print(data)
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'main-entropy-353108-c46118a2aa5c.json'
    # speech_client = SpeechClient() 
    # media_file_name_wav = data

    # with open(media_file_name_wav,'rb') as f1:
    #   byte_data = f1.read()
    # audio_wav = RecognitionAudio(content = byte_data)
    # config_wav = RecognitionConfig(
    #   audio_channel_count=1,
    #   enable_automatic_punctuation = True,
    #   language_code = 'en-US',
    # )
    # #recognize or longrunningrecognize
    # result_wav = speech_client.recognize(config = config_wav, audio=audio_wav)
    # print(result_wav)
    # complete_audio = ""
    # for result in result_wav.results:
    #   complete_audio += result.alternatives[0].transcript
    # result = get_summarization(complete_audio)
    result = get_summarization(data)
    print(result)
    return render(request, 'result.html', {"file_name": data, "summarized_text": result})
    #return HttpResponse(result)
    
    # For speech recognition using .wav file
    # r = sr.Recognizer()
    # with sr.AudioFile(data) as source:
    #     audio_text = r.record(source)

    #     try:
    #         text = r.recognize_google(audio_text)
    #         return HttpResponse(text)
    #     except Exception as e:
    #         print(e)
    #         return HttpResponse("Failed")