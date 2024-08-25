FROM python:3.9

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

ENV AZURE_TTS_KEY='b2079993b3e64e7f9631296aa9859349'
ENV AZURE_TTS_REGION='eastus'
ENV ELEVENLABS_API_KEY='5b6b4d89f1d8e0429416d971f994066b'
ENV OPENAI_API_KEY='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'
ENV GROQ_API_KEY='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'
ENV ANTHROPIC_API_KEY='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'
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]

