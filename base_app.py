"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image

# Data dependencies
import pandas as pd

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
	st.title("Global Tech Data Inc. Tweet Classifier")
	header_image = Image.open('resources/ourLogo.png')
	st.image(header_image, use_column_width='auto')
	st.subheader("Determine Users Climate Change Belief by classifying their tweets")


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Welcome","More Information","How to Use","Make a Prediction","About Us","Contact Us"]
	selection = st.sidebar.selectbox("Choose Option", options)
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
	if selection == "Make a Prediction":
		# Creating a text box for user input
		st.subheader("Make a Prediction")
		modelChoice = st.radio("Choose a model", ("Multi Logistic Regression", "Decision Tree", "Random Forest","Naive-Bayes","SVC(kernel=linear)","SVC(kernel=rbf)"))
		tweet_text = st.text_area("Enter a tweet below",placeholder="eg. RT @SoyNovioDeTodas: It's 2016, and a racist, sexist, climate change denying bigot is leading in the polls. #ElectionNight")
		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			if modelChoice == "Multi Logistic Regression":
				predictor = joblib.load(open(os.path.join("resources/LogisticRegression().pkl"), "rb"))
				prediction = predictor.predict(vect_text)
			elif modelChoice == "Decision Tree":
				predictor = joblib.load(open(os.path.join("resources/DecisionTreeClassifier(random_state=42).pkl"), "rb"))
				prediction = predictor.predict(vect_text)

			elif modelChoice == "Random Forest":
				predictor = joblib.load(open(os.path.join("resources/RandomForestClassifier(random_state=42).pkl"), "rb"))
				prediction = predictor.predict(vect_text)

			elif modelChoice == "Naive-Bayes":
				predictor = joblib.load(open(os.path.join("resources/MultinomialNB(alpha=0.1).pkl"), "rb"))
				prediction = predictor.predict(vect_text)

			elif modelChoice == "SVC(kernel=linear)":
				predictor = joblib.load(open(os.path.join("resources/SVC(kernel='linear').pkl"), "rb"))
				prediction = predictor.predict(vect_text)

			elif modelChoice == "SVC(kernel=rbf)":
				predictor = joblib.load(open(os.path.join("resources/SVC().pkl"), "rb"))
				prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	if selection == "How to Use":
		st.subheader("How to Use")
		st.markdown("All you have to do is follow these steps:")
		st.write("Step 1: Click the dropdown button on the left")
		st.write("Step 2: Choose the 'Make a Prediction' option")
		st.write("Step 3: Then choose a model from the available choices")
		st.write("Step 4: Enter a tweet in the text area")
		st.write("Step 5: Click Classify!")
	if selection == "About Us":
		st.subheader("About Us")
		st.write("Global Data Tech Inc. was founded by a handful of visionaries in 2022. The founders are Fabian Brijlal, William Hlatswayo, Sikelela Zungu, Nobuntu Mzilikazi, Juliet Bopape and Lindelwa Nhlapho. Their firm belief was that data was not just data, but rather information that holds life changing, world transforming potential. Data could be used to not just solve real world problems but create real world breakthroughs in areas where we didn't know we needed breakthroughs. But their vision takes it a step further. They want to make the insights gained from data easily accessible and understandable to everyone. This is why their motto is 'Simple solutions for complex connections.'")
	if selection == "Contact Us":
		st.subheader("Contact Us")
		st.write("Should you have any enquiries, please contact us via our email address: allfsocial@pm.me")
# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
	main()