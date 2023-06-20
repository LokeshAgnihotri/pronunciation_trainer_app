import json
import os
import random
from ast import literal_eval

from pydub import AudioSegment
import eng_to_ipa as ipa
import pyttsx3
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask import make_response
from flask_cors import CORS
from accuracy import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'
rootPath = ''

app.config['UPLOAD_FOLDER'] = 'reference_recordings/'

reference_recordings_path = 'reference_recordings/'
captured_recordings_path = 'captured_recordings/'

reference_text = ''


@app.route(rootPath + '/')
def main():
    return render_template('UI.html')


@app.route(rootPath + '/reference_recordings/', methods=['GET', 'POST'])
@app.route(rootPath + '/reference_recordings/<file_name>', methods=['GET', 'POST'])
def getAudio(file_name):
    """
    Retrieves an audio file and returns it as a response.

    Args:
        file_name (str): The name of the audio file to be retrieved.

    Returns:
        flask.Response: The response object containing the audio file.
    """
    try:
        audio_file_path = os.path.join(reference_recordings_path, file_name + ".mp3")

        if not os.path.isfile(audio_file_path):
            return make_response("Audio file not found", 404)

        with open(audio_file_path, 'rb') as file:
            response = make_response(file.read())
            response.headers['Content-Type'] = 'audio/mpeg'
            response.headers['Content-Disposition'] = 'attachment; filename=sound.mp3'
            return response

    except Exception as e:
        return make_response(f"An error occurred: {str(e)}", 500)


# speeech  to text

def convert_audio_to_text(audio_file_location_name):
    """
    Converts an audio file to text using speech recognition.

    Args:
        audio_file_location_name (str): The location and name of the audio file.

    Returns:
        str: The recognized text from the audio file.
    """
    converted_file = "captured_recordings/converted_audio.wav"

    try:
        # Convert audio file to PCM WAV format
        audio = AudioSegment.from_file(audio_file_location_name)
        audio.export(converted_file, format="wav")

        r = sr.Recognizer()

        # Load the converted audio file
        with sr.AudioFile(converted_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)

        # Remove the temporary converted file
        os.remove(converted_file)

        print(f"Recognized text: {text}")
        return text

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


#  A method to convert the text to speech using pyttsx3
def convert_text_to_speech(text, audio_name_location):
    """
    Converts the given text to speech and saves it as an audio file.

    Args:
        text (str): The text to be converted to speech.
        audio_name_location (str): The desired name and location of the output audio file.

    Returns:
        str: The filename of the generated audio file.
    """
    engine = pyttsx3.init()

    try:
        voices = engine.getProperty('voices')
        voice = None

        # Search for a female voice
        for v in voices:
            if v.gender == 'female':
                voice = v
                break

        if voice is None:
            # If no female voice found, use the default voice
            voice = voices[0]

        engine.setProperty('voice', voice.id)
        pronunciation_audio_file = f"{audio_name_location}.mp3"
        engine.save_to_file(text, pronunciation_audio_file)
        engine.runAndWait()

        print(f"Text: {text}")
        print(f"Audio file saved as: {pronunciation_audio_file}")
        return pronunciation_audio_file

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


# convert text to IPA
def convert_text_to_ipa(text):
    return ipa.convert(text)


# Send audio and phonetic transcription to the front end
@app.route(rootPath + '/pronunciation_trainer', methods=['POST'])
def pronunciation_trainer():
    event = {'body': json.dumps(request.get_json(force=True))}
    text = event['body']
    # text = request.json['text']
    phenome = convert_text_to_ipa(text)

    #   Generate the audio file for the pronunciation

    ref_audio_save_path = os.path.join(reference_recordings_path)
    if not os.path.exists(ref_audio_save_path):
        os.makedirs(ref_audio_save_path)

    ref_audio_file_save_path = os.path.join(ref_audio_save_path, literal_eval(text).replace(' ', '_'))
    sound = convert_text_to_speech(text, ref_audio_file_save_path)

    #   Create JSON response
    response = {
        'phenome': phenome,
        'sound': sound
    }
    return json.dumps(response)


@app.route(rootPath + '/receiver', methods=['POST'])
def getAudioFromText():
    event = {'body': json.dumps(request.get_json(force=True))}
    print(event)
    data = request.get_json()
    data = jsonify(data)
    return data


# Suggest next word
@app.route('/next_word')
def random_word():
    with open('english_dictionary.txt', 'r') as file:
        words = file.read().splitlines()
    selected_word = random.choice(words)
    random_word_ipa = ipa.convert(selected_word)

    # Generate the audio file for the pronunciation
    ref_audio_save_path = os.path.join(reference_recordings_path)
    if not os.path.exists(ref_audio_save_path):
        os.makedirs(ref_audio_save_path)

    ref_audio_file_save_path = os.path.join(ref_audio_save_path, selected_word.replace(' ', '_'))
    sound = convert_text_to_speech(selected_word, ref_audio_file_save_path)

    return jsonify({'random_word': selected_word, 'random_word_ipa': random_word_ipa,
                    'pronunciation_audio': ref_audio_file_save_path})


# https://www.makeuseof.com/tag/python-javascript-communicate-json/


# route to receive the audioop
@app.route('/upload-audio', methods=['POST', 'GET'])
def upload_audio():
    if 'audio' not in request.files:
        return "error there is no audio"
    if 'ref_text' not in request.form:
        return "no text received"

    audio_file = request.files['audio']
    save_path = os.path.join(captured_recordings_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, 'audio.wav')
    audio_file.save(file_path)
    text = convert_audio_to_text(file_path)
    print(jsonify({'text': text}))

    ref_text = request.form['ref_text']

    word_error = wer(ref_text, text)
    char_error = cer(ref_text, text)

    real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices \
        = matchSampleAndRecordedWords(ref_text, text)

    pronunciation_accuracy, current_words_pronunciation_accuracy \
        = getPronunciationAccuracy(real_and_transcribed_words)

    return jsonify({'word_error': word_error, 'char_error': char_error,
                    'pronunciation_accuracy': pronunciation_accuracy,
                    'current_words_pronunciation_accuracy': current_words_pronunciation_accuracy,
                    'real_and_transcribed_words': real_and_transcribed_words,
                    'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa})


def get_accuracy(ref_text, actual_text):
    # Accuracy...
    correct_letter = 0
    total_letters = 0
    for char in ref_text:
        if char.isalpha():
            total_letters += 1
    ref_text_array = ref_text.split()
    spoken_text_array = actual_text.split()
    for i in range(0, len(ref_text_array)):
        if ref_text_array[i] != spoken_text_array[i]:
            if len(ref_text_array) == len(spoken_text_array):
                for letter1, letter2 in zip(ref_text_array[i], spoken_text_array[i]):
                    if letter1 == letter2:
                        correct_letter += 1
    return correct_letter / total_letters


if __name__ == '__main__':
    language = 'en'
    app.run(host="0.0.0.0", port=5000, debug=True)
