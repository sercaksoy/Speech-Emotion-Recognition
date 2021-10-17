FROM python:3.8.8

EXPOSE 8501

ADD main.py audio_func.py ./

COPY requirements.txt .
COPY models/ models/

RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

CMD ["streamlit", "run", "main.py"]