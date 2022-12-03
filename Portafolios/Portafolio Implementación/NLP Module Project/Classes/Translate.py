import os
import json
import requests
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


class Translator:
    load_dotenv()

    def __init__(self, data_en, data_es):
        self.API_KEY = os.environ.get("X_RAPID_API_KEY")
        data_en = open(data_en, "r", encoding="utf8")
        self.data_en = data_en.readlines()
        data_es = open(data_es, "r", encoding="utf8")
        self.data_es = data_es.readlines()
        data_es.close()
        data_en.close()

    def bleu(self, original, translation):
        ref = []
        for sentence in original:
            ref.append(sentence.split())
        preprocessed_translation = []
        for sentence in translation:
            preprocessed_translation.append(sentence.split())
        bleu_scores = []
        for sentence in preprocessed_translation:
            bleu_score = sentence_bleu(ref, sentence)
            bleu_scores.append(bleu_score)
        return bleu_scores

    def translate_plus(self, text="Hello world!", source="en", target="es"):
        url = "https://translate-plus.p.rapidapi.com/translate"
        payload = {"text": text, "target": target, "source": source}
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": self.API_KEY,
            "X-RapidAPI-Host": "translate-plus.p.rapidapi.com",
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        response = json.loads(response.text)["translations"]["translation"]
        return response

    def multi_translate(self, text=None, source="en", target="es"):
        if text is None:
            text = ["Hello World"]

        url = "https://rapid-translate-multi-traduction.p.rapidapi.com/t"

        payload = {"from": source, "to": target, "e": "", "q": text}
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": "4916ad8781msh837ffe6b021be08p1f3f94jsnb9bb7f775911",
            "X-RapidAPI-Host": "rapid-translate-multi-traduction.p.rapidapi.com",
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        # The API sends a string wrong formatted instead of a JSON, so we have to delete all the '"' characters
        # And then we have to split the string using the ',' character to create the formatted JSONs.
        response = response.text.replace('"', "")[1:-1].split(",")
        return response

    def score_model(self):
        model_1 = []
        print("I am translating, please wait...")
        for i in range(len(self.data_en)):
            self.data_en[i] = self.data_en[i].strip("\n")
            self.data_es[i] = self.data_es[i].strip("\n")

        for i in range(len(self.data_en)):
            translated = self.translate_plus(self.data_en[i])
            model_1.append(translated)

        model_2 = self.multi_translate(self.data_en)
        model_1_score = self.bleu(self.data_es, model_1)
        model_2_score = self.bleu(self.data_es, model_2)
        m1_score = np.average(model_1_score)
        m2_score = np.average(model_2_score)
        print(f"M1: {m1_score} \nM2:{m2_score}")


if __name__ == "__main__":
    t = Translator()
    t.translate_plus()
    t.multi_translate(
        ["Hello World", "Hi, my name is Erick", "Good afternoon", "This is a test"]
    )
