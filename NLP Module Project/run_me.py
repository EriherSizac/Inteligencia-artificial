from Classes.SentimentAnalysis import SentimentAnalysis
from Classes.Ner import Ner
from Classes.Translate import Translator

print("======================= TESTING SENTIMENT ANALYSIS ===========================")
SA = SentimentAnalysis()
print("-------------- PREDICTIONS --------------------")
SA.predict("./datasets/tiny_movie_reviews_dataset.txt")
input("Press ENTER to continue to next model")

print()
print("======================= Ner ===========================")
print("Do not forget to change sample size in .env file.")
ner = Ner()
ner.train()
ner.plot()
input("Press ENTER to continue to next model")

print()
print("=================== TRANSLATE BLEU SCORE ==================")
t = Translator("./datasets/europarl-v7.es-en.en", "./datasets/europarl-v7.es-en.es")
t.score_model()
