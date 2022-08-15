# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Data dependencies
import pandas as pd

def remove_links_and_usernames(tweet):
	tweet = tweet.lower()
	tweet = re.sub(r'http\S+','', tweet)
	tweet = re.sub(r'@\S+','',tweet)
	return tweet

def remove_stop_words(tweet):
	sw = stopwords.words('english')
	sw.append('USERNAME')
	sw.append('URL')
	sw.append('rt')
	tweetfour = [y for y in tweet if not y in sw]
	return tweet


def remove_punctuation(tweet):
	tweet = re.sub(r'[^\w\s]', '', str(tweet))
	tweet = tweet.lstrip()
	tweet = tweet.rstrip()
	tweet = tweet.replace('  ', ' ')
	return tweet

# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

# Vectorizer
news_vectorizer = open("resources/fittedVect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """
	# Creates a main title and subheader on your page -
	# these are static across all pages
	jumbotron = Image.open('resources/jumbotron.jpeg')
	st.image(jumbotron, use_column_width='auto')
	st.subheader("Global Tech Data Inc. Tweet Classifier")
	# header_image = Image.open('resources/ourLogo.png')
	# st.image(header_image, use_column_width='auto')
	# st.subheader("Determine Users Climate Change Belief by classifying their tweets")
	t_choices = {2: "linking to factual news about climate change",
			   1: "supporting the belief of man-made climate change",
			   0: "neither supporting nor refuting the belief of man-made climate change",
			   -1: "as not not believing in man-made climate change"}

	# Prediction Labels
	prediction_labels = {'Anti Climate Change': -1,
						'Neutral about Climate Change': 0,
						'Pro Climate Change': 1,
						'News about Climate Change': 2}
	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Welcome","More Information","How to Use","Try other Models","About Us","Contact Us"]
	selection = st.sidebar.selectbox("Navigation Menu", options)
#####################
	# Building out the "Welcome" page
	if selection == "Welcome":
		# Welcome test for the landing page
		st.write(
		"""
		Do you care about **Climate Change**? Here's a climate change tweet sentiment classifier.

		Try it right Now! We'll predict your climate change sentiment based on your tweet.
		Or it could simply be News! Try it out below.
		"""
		)
		# Creating a text box for user input
		st.write("Default Prediciton Model: **Support Vector Classifier**")
		welcome_tweet_text = st.text_area("Enter a tweet below",
								placeholder="eg. RT @SoyNovioDeTodas: It's 2016, and a racist, sexist, climate change denying bigot is leading in the polls. #ElectionNight")

		links_removed = remove_links_and_usernames(welcome_tweet_text)
		sw_removed = remove_stop_words(links_removed)
		pun_removed = remove_punctuation(sw_removed)
		vect_text = tweet_cv.transform([pun_removed]).toarray()

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']])  # will write the df to the page

		if st.button("Classify"):
			predictor = joblib.load(open(os.path.join("resources/SVC().pkl"), "rb"))
			prediction = predictor.predict(vect_text)
			final_result = get_key(prediction,prediction_labels)
			st.success("Tweet classified as {}".format(final_result))

#####################
	# Building out the "Information" page
	if selection == "More Information":
		# You can read a markdown file from supporting resources folder
		st.subheader("More information")
		st.markdown("So you are probably wondering how this works. Well you see long ago when Merlin was alive and Harry Potter-We're just joking! This app which was made by our hard working team, allows you to enter a tweet. Our algorithms will then run some calculations and math gymnastics in the background to determine whether that tweet is written by someone who believes or denies Climate Change.")
		st.markdown("We are aware that the world is not just black and white and so we have varying degrees of belief/disbelief. So here is a breakdown of what the outputs you may be getting mean:")
		mycol1,mycol3 = st.columns(2)
		with mycol1:
			st.success("2")
			st.info("1")
			st.warning("0")
			st.error("-1")
		with mycol3:
			st.markdown("News: the tweet links to factual news about climate change")
			st.markdown("Pro: the tweet supports the belief of man-made climate change")
			st.markdown("Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
			st.markdown("Anti: the tweet does not believe in man-made climate change")
#####################
	# Building out the predication page
	if selection == "Try other Models":
		# Creating a text box for user input
		st.subheader("Make a Prediction")
		modelChoice = st.selectbox(on_change=None,
									label='Please choose a model',
									options=("Multi Logistic Regression","Decision Tree","Random Forest","Naive-Bayes","SVC(kernel=linear)","SVC(kernel=rbf)"))
		#modelChoice = st.radio("Choose a model", ("Multi Logistic Regression","Decision Tree","Random Forest","Naive-Bayes","SVC(kernel=linear)","SVC(kernel=rbf)"))
		tweet_text = st.text_area("Enter a tweet",
									placeholder="eg. RT @SoyNovioDeTodas: It's 2016, and a racist, sexist, climate change denying bigot is leading in the polls. #ElectionNight")


		if modelChoice == "Multi Logistic Regression":
			# Transforming user input with vectorizer
			links_removed = remove_links_and_usernames(tweet_text)
			sw_removed = remove_stop_words(links_removed)
			pun_removed = remove_punctuation(sw_removed)
			vect_text = tweet_cv.transform([pun_removed]).toarray()
			if st.button("Classify"):
				predictor = joblib.load(open(os.path.join("resources/LogisticRegression().pkl"), "rb"))
				prediction = predictor.predict(vect_text)
				final_result = get_key(prediction,prediction_labels)
				st.success("Tweet classified as {}".format(final_result))


		elif modelChoice == "Decision Tree":
			links_removed = remove_links_and_usernames(tweet_text)
			sw_removed = remove_stop_words(links_removed)
			pun_removed = remove_punctuation(sw_removed)
			vect_text = tweet_cv.transform([pun_removed]).toarray()
			if st.button("Classify"):
				predictor = joblib.load(open(os.path.join("resources/DecisionTreeClassifier(random_state=42).pkl"), "rb"))
				prediction = predictor.predict(vect_text)
				final_result = get_key(prediction,prediction_labels)
				st.success("Tweet classified as {}".format(final_result))

		elif modelChoice == "Random Forest":
			links_removed = remove_links_and_usernames(tweet_text)
			sw_removed = remove_stop_words(links_removed)
			pun_removed = remove_punctuation(sw_removed)
			vect_text = tweet_cv.transform([pun_removed]).toarray()
			if st.button("Classify"):
				predictor = joblib.load(open(os.path.join("resources/RandomForestClassifier(random_state=42).pkl"), "rb"))
				prediction = predictor.predict(vect_text)
				final_result = get_key(prediction,prediction_labels)
				st.success("Tweet classified as {}".format(final_result))

		elif modelChoice == "Naive-Bayes":
			links_removed = remove_links_and_usernames(tweet_text)
			sw_removed = remove_stop_words(links_removed)
			pun_removed = remove_punctuation(sw_removed)
			vect_text = tweet_cv.transform([pun_removed]).toarray()
			if st.button("Classify"):
				predictor = joblib.load(open(os.path.join("resources/MultinomialNB(alpha=0.1).pkl"), "rb"))
				prediction = predictor.predict(vect_text)
				final_result = get_key(prediction,prediction_labels)
				st.success("Tweet classified as {}".format(final_result))

		elif modelChoice == "SVC(kernel=linear)":
			links_removed = remove_links_and_usernames(tweet_text)
			sw_removed = remove_stop_words(links_removed)
			pun_removed = remove_punctuation(sw_removed)
			vect_text = tweet_cv.transform([pun_removed]).toarray()
			if st.button("Classify"):
				predictor = joblib.load(open(os.path.join("resources/SVC(kernel='linear').pkl"), "rb"))
				prediction = predictor.predict(vect_text)
				final_result = get_key(prediction,prediction_labels)
				st.success("Tweet classified as {}".format(final_result))

		elif modelChoice == "SVC(kernel=rbf)":
			links_removed = remove_links_and_usernames(tweet_text)
			sw_removed = remove_stop_words(links_removed)
			pun_removed = remove_punctuation(sw_removed)
			vect_text = tweet_cv.transform([pun_removed]).toarray()
			if st.button("Classify"):
				predictor = joblib.load(open(os.path.join("resources/SVC().pkl"), "rb"))
				prediction = predictor.predict(vect_text)
				final_result = get_key(prediction,prediction_labels)
				st.success("Tweet classified as {}".format(final_result))

########################
	if selection == "How to Use":
		st.subheader("How to Use")
		st.markdown("All you have to do is follow these steps:")
		st.write("Step 1: Click the dropdown button on the left")
		st.write("Step 2: Choose the 'Make a Prediction' option")
		st.write("Step 3: Then choose a model from the available choices")
		st.write("Step 4: Enter a tweet in the text area")
		st.write("Step 5: Click Classify!")
#########################
	if selection == "About Us":
		st.subheader("About Us")
		st.write("Global Data Tech Inc. was founded by a handful of visionaries in 2022. The founders are Fabian Brijlal, William Hlatswayo, Sikelela Zungu, Nobuntu Mzilikazi, Juliet Bopape and Lindelwa Nhlapho. Their firm belief was that data was not just data, but rather information that holds life changing, world transforming potential. Data could be used to not just solve real world problems but create real world breakthroughs in areas where we didn't know we needed breakthroughs. But their vision takes it a step further. They want to make the insights gained from data easily accessible and understandable to everyone. This is why their motto is 'Simple solutions for complex connections.'")
		fabians_pic = Image.open('resources/imgs/fabian.png')
		juliet_pic = Image.open('resources/imgs/juliet.png')
		nobuntu_pic = Image.open('resources/imgs/nobuntu.png')
		sikel_pic = Image.open('resources/imgs/sikelela.png')
		will_pic = Image.open('resources/imgs/william.png')
		char_pic = Image.open('resources/imgs/charmain.png')
		c1,c2,c3 = st.columns(3)
		with c1:
			st.image(fabians_pic, caption="Fabian Brijlal(Director)",use_column_width='always')
			st.image(juliet_pic, caption="Juliet Bopape(Senior Data Analyst)",use_column_width='always')
		with c2:
			st.write()
			st.image(will_pic,caption="William Hlatshwayo(Director)",use_column_width='always')
			st.image(sikel_pic,caption="Sikelela Zungu(Data Analyst)",use_column_width='always')
		with c3:
			st.image(char_pic,caption="Lindelwa Nhlapho(Data Analyst)",use_column_width='always')
			st.image(nobuntu_pic, caption="Nobuntu Mzilikazi(Data Scientist)",use_column_width='always')

	if selection == "Contact Us":
		st.subheader("Contact Us")
		st.write("Should you have any enquiries, please contact us via our email address: enquiries@globaldatatechinc.com")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()
