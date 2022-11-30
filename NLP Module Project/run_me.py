from Classes.SentimentAnalysis import SentimentAnalysis
from Classes.Ner import Ner
from Classes.Translate import Translator

print("======================= TESTING SENTIMENT ANALYSIS ===========================")
SentimentModel = SentimentAnalysis()
print("-------------- PREDICTIONS --------------------")
SentimentModel.predict("./datasets/tiny_movie_reviews_dataset.txt")
input("Press ENTER to continue to next model")

print()
print("======================= Ner ===========================")
print("Do not forget to change sample size in .env file.")
ner_model = Ner()
ner_model.train()
ner_model.plot()
input("Press ENTER to continue to next model")

print()
print("=================== TRANSLATE BLEU SCORE ==================")
translator = Translator("./datasets/europarl-v7.es-en.en", "./datasets/europarl-v7.es-en.es")
translator.score_model()
