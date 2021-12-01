#Load all required libraries

from flask import Flask, jsonify, request, send_file, render_template
from flask_cors import CORS, cross_origin
import requests
import time
import traceback
import pandas as pd
import string


#Initialize the flask application
app = Flask(__name__)
cors = CORS(app)


#Preload all files required for recommendations
df_movie_summary=pd.read_csv('df_movie_summary.csv')

#Load df_p file which has the names of titles present together in previously watched. 
df_p=pd.read_csv('df_p.csv')
df_p = pd.pivot_table(df_p,index='Cust_Id')

#Load movie titles csv.
df_title = pd.read_csv('movie_titles.csv', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)

def recommend(movie_title, min_count):   #Finding recommendations for movies with Pearson Correlation
	
	#Find index of input movie in the movie_titles.csv
	i = int(df_title.index[df_title['Name'] == movie_title][0])

	#Find the corresponding row of data in the df_p.csv
	target = df_p[str(i)]

	#Find correlation of all movies with the target
	similar_to_target = df_p.corrwith(target)

	#Create dataframe of movies with their corresponding correlation values
	corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
	corr_target.dropna(inplace = True)

	#Sort values in descending order of correlation
	corr_target = corr_target.sort_values('PearsonR', ascending = False)
	corr_target.index = corr_target.index.map(int)
	corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]

	#Return top 10 recommendations
	return(corr_target[corr_target['count']>min_count][1:11])

@app.route('/') #Landing Page API
@cross_origin()
def main():
	#Sends main.html as the landing page for the website when http://localhost:5000 is loaded
	return send_file('main.html')

@app.route('/recommendations', methods=['POST']) #Connection to Recommendation Page
@cross_origin()
def hello():
	#The movie that is clicked is sent from the frontend to this API. The value sent will have the ID along with the coordinates. Ex: "Batman Begins.x". We separate the .x and take the movie name.
	for key, val in request.form.items():
		if ".x" in key:
			movie_name=key[:key.index(".x")]
			break
	#Send the name of the movie to the function recommender above in order to generate recommendations.
	a=list(recommend(movie_name, 0)["Name"])

	#Split the recommendations into two lists for displaying on the website.
	img1=[]
	img2=[]
	for i in range(5):
		img1.append("static/img/posters/"+a[i].translate(str.maketrans('', '', string.punctuation)).upper().replace(" ","_")+".jpg")
	for i in range(5,10):
		img2.append("static/img/posters/"+a[i].translate(str.maketrans('', '', string.punctuation)).upper().replace(" ","_")+".jpg")
	
	#Send the list of recommendations to final_page.html for display.
	return render_template('final_page.html',img1=img1,img2=img2)

#Set the application to run on the url 0.0.0.0 with port=5000. Debug=True enables debug commands and output.
if __name__ == "__main__":
	app.run(host='0.0.0.0',port=5000,debug=True)