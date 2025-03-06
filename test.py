from chatbot import Chatbot

def test_sentiment_words():
    # Create a chatbot instance
    bot = Chatbot()
    
    # Test individual words
    test_words = ["enjoy", "like"]
    
    # Query each word and print the sentiment
    for word in test_words:
        sentiment = bot.calc_word_sentiment(word)
        sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        print(f"The word '{word}' has sentiment: {sentiment} ({sentiment_label})")
    
    # Test extract_sentiment on a preprocessed sentence
    test_sentence = "I didn't enjoy " + '"Titanic (1997)".'
    preprocessed = bot.preprocess(test_sentence)
    sentiment = bot.extract_sentiment(preprocessed)
    sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
    
    print(f"\nSentence: '{test_sentence}'")
    print(f"Preprocessed: '{preprocessed}'")
    print(f"Sentiment: {sentiment} ({sentiment_label})")

if __name__ == "__main__":
    test_sentiment_words()