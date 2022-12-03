from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class SentimentAnalysis:

    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("acho0057/sentiment_analysis_custom")
        model = AutoModelForSequenceClassification.from_pretrained("acho0057/sentiment_analysis_custom")

        self.pipeline = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

    def predict(self, path):
        from Classes.File import File
        dataset = File(path).lines
        output = self.pipeline(dataset)
        for res in output:
            print(res['label'].upper())


if __name__ == '__main__':

    sentiment = SentimentAnalysis()
    sentiment.predict("../datasets/tiny_movie_reviews_dataset.txt")
