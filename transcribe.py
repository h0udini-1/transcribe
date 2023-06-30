import io
import os
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import openai
import pyaudio
import tiktoken
import wavio
import concurrent.futures
import numpy as np
import tempfile
import queue
import threading
from datetime import datetime
import tiktoken
from tkinter import Scrollbar

OPEN_SANS_FONT = ("Open Sans", 12)

class TranscriptionApp:
    RECORD_SECONDS = 6 
    def __init__(self, window):
        self.window = window
        self.filename = None
        self.recording = False

        self.window.config(bg="#467589")

        self.scrollbar = ttk.Scrollbar(window)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(window, bg="#467589", yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.canvas.yview)

        self.container = tk.Frame(self.canvas, bg="#467589")
        self.canvas.create_window((0, 0), window=self.container, anchor=tk.NW)

        self.container.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.upload_button = tk.Button(self.container, text="Upload File/Încărcați Audio", command=self.upload_file, font=OPEN_SANS_FONT, bg="#184156", fg="#F8F8F8")
        self.upload_button.pack(pady=10)
        
        self.file_label = tk.Label(window, text="No file uploaded/Nu a fost încărcat audio", font=OPEN_SANS_FONT, bg="#467589", fg="#F8F8F8")
        self.file_label.pack(pady=5)

        self.transcribe_button = tk.Button(window, text="Transcribe/Transcrie", command=self.transcribe, state=tk.DISABLED, font=OPEN_SANS_FONT, bg="#184156", fg="#F8F8F8")
        self.transcribe_button.pack(pady=10)

        self.translate_button = tk.Button(window, text="Translate to English/Traduce in engleza", command=self.translate, state=tk.DISABLED, font=OPEN_SANS_FONT, bg="#184156", fg="#F8F8F8")
        self.translate_button.pack(pady=10)

        self.live_transcribe_button = tk.Button(window, text="Live Transcribe/Transcriere live", command=self.live_transcribe, font=OPEN_SANS_FONT, bg="#184156", fg="#F8F8F8")
        self.live_transcribe_button.pack(pady=10)

        self.transcription_display = tk.Text(window, font=OPEN_SANS_FONT, bg="#F8F9F8", fg="#161518")
        self.transcription_display.pack(pady=10)

        self.blinking_circle = tk.Canvas(window, width=20, height=20)
        self.blinking_circle.pack()
        self.red_circle = self.blinking_circle.create_oval(5, 5, 15, 15, fill='red')
        self.blinking_circle.itemconfig(self.red_circle, state='hidden')  # Initially hidden

       
        self.status_label = tk.Label(window, text="", font=("Open Sans", 10), bg="#467589", fg="#F8F8F8")
        self.status_label.pack(pady=5)

        self.frames_queue = queue.Queue()
        self.transcription_thread = threading.Thread(target=self.transcription_worker)
        self.transcription_thread.start()

        
        self.summarize_button = tk.Button(window, text="Summarize/Sumarizează", command=self.summarize, state=tk.ACTIVE, font=OPEN_SANS_FONT, bg="#184156", fg="#F8F8F8")
        self.summarize_button.pack(pady=10)
        
        self.summary_frame = tk.Frame(self.container, bg="#467589")
        self.summary_frame.pack(pady=10)

        self.summary_display = tk.Text(self.summary_frame, font=OPEN_SANS_FONT, bg="#F8F9F8", fg="#161518")
        self.summary_display.pack()
        
        self.summary_display = tk.Text(window, font=OPEN_SANS_FONT, bg="#F8F9F8", fg="#161518")
        self.summary_display.pack(pady=10)
        self.summary_display.pack_forget()  
        
        self.close_summary_button = tk.Button(window, text="Close Summary/Închideți rezumatul", command=self.close_summary, font=OPEN_SANS_FONT, bg="#184156", fg="#F8F8F8")
        self.close_summary_button.pack(pady=10)
        self.close_summary_button.pack_forget() 


    def transcribe(self):
        self.status_label.config(text="Transcribing audio file/Transcrierea fișierului audio...")
        self.window.update()
        transcription = self.transcribe_audio_file(self.filename)
        print(transcription)
        reformatted_transcription = self.process_text(str(transcription))
        self.transcription_display.delete('1.0', tk.END)
        self.transcription_display.insert(tk.END, reformatted_transcription)
        # print(reformatted_transcription)
        self.status_label.config(text="Transcription completed!/Transcrierea finalizată!")

    def live_transcribe(self):
        if not self.recording:
            self.recording = True
            self.live_transcribe_button.config(text="Stop Transcription/Opriți transcrierea")
            self.status_label.config(text="Live transcription in progress/Transcriere live în curs...")
            self.blinking_circle.itemconfig(self.red_circle, state='normal')  # Show red circle
            self.blink_circle()  # Start blinking
            self.start_live_transcription()
        else:
            self.recording = False
            self.live_transcribe_button.config(text="Live Transcribe/Transcriere live")
            self.status_label.config(text="Live transcription stopped/Transcrierea live a fost oprită.")
            self.blinking_circle.itemconfig(self.red_circle, state='hidden')  # Hide red circle
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.frames_queue.put(None)
            # Reset the queue and frames
            self.frames_queue = queue.Queue()
            self.frames = []
            # Create a new transcription thread
            self.transcription_thread = threading.Thread(target=self.transcription_worker)
            self.transcription_thread.start()
            # Show the summarize button
            self.summarize_button.config(state=tk.NORMAL)
            self.summarize_button.pack()

    def blink_circle(self):
        if self.recording:
            current_state = self.blinking_circle.itemcget(self.red_circle, 'state')
            new_state = 'hidden' if current_state == 'normal' else 'normal'
            self.blinking_circle.itemconfig(self.red_circle, state=new_state)
            # Schedule next blink in 500 ms 
            self.window.after(500, self.blink_circle)
    
    def upload_file(self):
        self.filename = filedialog.askopenfilename()
        if self.filename:
            self.transcribe_button.config(state=tk.NORMAL, bg="#B74C40")
            self.translate_button.config(state=tk.NORMAL, bg="#B74C40")

            # Update file display label
            self.file_label.config(text=f"File: {self.filename}")

    def translate(self):
        transcription = self.translate_audio_file(self.filename)
        reformatted_transcription = self.process_text(self, transcription)
        self.transcription_display.delete('1.0', tk.END)  
        self.transcription_display.insert(tk.END, reformatted_transcription)

   
    
  
    def process_text(self, text):
        # Initializing OpenAI API key
        openai.api_key = "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"

        # Creating the prompt for the OpenAI model
        prompt_base = "Make this text more readable by skipping 2 lines where a speaker of the text would normally take a pause. Output exactly the same words but with new lines in those places.\n\nText:\n"

        # Breaking the input text into chunks that can be processed by the OpenAI model
        token_chunks = self.get_token_chunks(text)

        # Initializing a variable to store the reformatted text
        reformatted_text = ""

        # Using multithreading to process the chunks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunk in token_chunks:
                # Adding the chunk to the prompt
                prompt = f"{prompt_base}{chunk}"

                # Getting the model's response
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature = 0.2,
                    messages=[
                        {"role": "system", "content": "You are the best text formatter for transcripts of spoken audio"},
                        {"role": "user", "content": prompt}
                    ]
                    )
                print(response.choices[0].message)
                # Appending the model's response to the reformatted text
                reformatted_text += response.choices[0].message['content'].strip()

        return reformatted_text


    

    @staticmethod
    def get_token_chunks(text: str, max_tokens: int = 3000):
        encoding = tiktoken.get_encoding("cl100k_base")
        print(f"Type of text: {type(text)}")
        print(f"Value of text: {text}")
        # Convert the text into a list of tokens
        tokens = encoding.encode(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            # Get token string representation
            token_string = encoding.decode([token])
            
            if current_length + len(token_string.split()) > max_tokens and current_chunk:
                # Convert current chunk of tokens back into text and append it to the chunks list
                chunks.append(''.join([encoding.decode([token]) for token in current_chunk]))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(token)
            current_length += len(token_string.split())

        # If there is any remaining chunk, convert it back into text and append it to the chunks list
        if current_chunk:
            chunks.append(''.join([encoding.decode([token]) for token in current_chunk]))

        return chunks


    
    def transcribe_audio_file(self, audio_file):
        openai.api_key = "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"
            
        try:
            if isinstance(audio_file, str):  # If the audio_file is a file path
                with open(audio_file, "rb") as f:
                    response = openai.Audio.transcribe("whisper-1", f)
            else:  # If the audio_file is a BytesIO object
                # Write the BytesIO stream to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(audio_file.read())
                # Transcribe the temporary file
                with open(tmp.name, "rb") as f:
                    response = openai.Audio.transcribe("whisper-1", f)
                os.unlink(tmp.name)  # Delete the file manually
            print(response)
            return response
        except Exception as e:
            print("An error occurred during transcription:")
            print(e)
            

    
    def translate_audio_file(file_path):
        openai.api_key = "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"
        with open(file_path, "rb") as audio_file:
            response = openai.Audio.translate("whisper-1", audio_file)
        return response['text']
    
    def live_transcribe(self):
        if not self.recording:
            self.recording = True
            self.live_transcribe_button.config(text="Stop Transcription/Opriți transcrierea")
            self.status_label.config(text="Live transcription in progress/Transcriere live în curs...")
            self.blinking_circle.itemconfig(self.red_circle, state='normal')  # Show red circle
            self.blink_circle()  # Start blinking
            self.start_live_transcription()
        else:
            self.recording = False
            self.live_transcribe_button.config(text="Live Transcribe/Transcriere live")
            self.status_label.config(text="Live transcription stopped/Transcrierea live a fost oprită.")
            self.blinking_circle.itemconfig(self.red_circle, state='hidden')  # Hide red circle
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.frames_queue.put(None)
            # Reset the queue and frames
            self.frames_queue = queue.Queue()
            self.frames = []
            # Create a new transcription thread
            self.transcription_thread = threading.Thread(target=self.transcription_worker)
            self.transcription_thread.start()

    def start_live_transcription(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        self.frames = []

        self.audio = pyaudio.PyAudio()

        def callback(in_data, frame_count, time_info, status):
            if self.recording:
                self.frames.append(in_data)
            return in_data, pyaudio.paContinue

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=callback
        )

        self.stream.start_stream()

        start_time = time.time()

        while self.recording:
            self.window.update()
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.RECORD_SECONDS:
                start_time = time.time()
                self.frames_queue.put(self.frames)
                self.frames = []
        # Signal to the worker that the recording has ended
        self.frames_queue.put(None)

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        

    def transcribe_live_audio(self, frames):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        recorded_audio = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            wavio.write(temp_audio_file.name, audio_array, RATE, sampwidth=2)
            temp_audio_file.close()  # Ensure the file is closed before opening it again

            # Transcribe the audio file
            transcription_response = self.transcribe_audio_file(temp_audio_file.name)
            transcript = ""  # Initialize transcript
            try:
                print(f"Before transcription at {datetime.fromtimestamp(time.time())}: {transcription_response}")
                transcript = transcription_response["text"]
                print(f"After transcription: {transcript}")
                self.output_words_live(transcript)  # Output words one by one
                # Commented out the line that output the whole transcription at once
                # self.transcription_display.insert(tk.END, "\n" + transcript)
            except Exception as e:
                print("An exception occurred:")
                print(e)

        os.unlink(temp_audio_file.name)  # Delete the file manually

    def transcription_worker(self):
        while True:
            frames = self.frames_queue.get()
            if frames is None:  # End of recording
                break
            self.transcribe_live_audio(frames)

    def output_words_live(self, text):
        words = text.split()
        num_words = len(words)
        time_interval = self.RECORD_SECONDS / num_words
        for word in words:
            print(word)
            self.transcription_display.insert(tk.END, word + " ")
            self.window.update()
            time.sleep(time_interval)

    def summarize(self):
        self.status_label.config(text="Summarizing the transcript/Sumarizând transcrierea...")
        self.window.update()
        transcript = self.transcription_display.get('1.0', tk.END).strip()
        summary = self.create_summary(transcript)
        self.summary_display.delete('1.0', tk.END)
        self.summary_display.insert(tk.END, summary)
        self.status_label.config(text="Summary completed!/Rezumatul finalizat!")
        self.summary_display.pack()  # Show the summary display
        self.close_summary_button.pack()  # Show the close summary button

    @staticmethod
    def create_summary(text):
        openai.api_key = "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"
        prompt = f"Create a detailed point form summary with headings to display the main ideas and supporting info from this transcript:\n{text}"
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=3000)
        return response.choices[0].text.strip()

    def close_summary(self):
        self.summary_display.pack_forget()  # Hide the summary display
        self.close_summary_button.pack_forget()  # Hide the close summary button
        self.summarize_button.pack_forget()  # Hide the summarize button

if __name__ == "__main__":
    window = tk.Tk()
    app = TranscriptionApp(window)
    window.mainloop()
