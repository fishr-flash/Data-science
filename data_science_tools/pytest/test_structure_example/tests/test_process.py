from src.process import extract_sentiment


def test_extract_sentiment():
	text = "Today I found a duck and I am happy"

	sentiment = extract_sentiment(text)

	assert sentiment > 0
