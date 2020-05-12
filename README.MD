<h1>Using Ridge classifier with Board Game Geek Review Data to design a model and then host this model on a web server using Flask.</h1>

<h2>You can see the actual hosting implementation by clicking on <a href = "https://myridgeclassifierlive.herokuapp.com/">this</a></h2>

To deploy this project on your own local server or on another hosting site, you need to perform the following steps:
1. Create a Folder in your Local Directory using mkdir in command prompt
2. After creating your work directory, enter the directory through command prompt using cd <directory_name>
3. Once we are in the work directory, we first need to install python if you do not have it already.
4. After downloading python, we need to create a virtual environment in our work directory. We first install it by using "pip3 install virtualenv"
5. We can do that using "virtualenv venv" command in command prompt
6. After running that command , we can activate the virtual environment using "venv\scripts\activate"
7. Once we are in our virtual environment, we have to install flask. On windows you can do that by typing "pip3 install flask"
8. After flask is downloaded, we can open our python code. After entering all the code for our HTML and our Python backend, we now move on to deployment.
9. Another important thing is that we need to have git installed before we can use heroku to deploy our web application.
10. We first initaite git, add our files then commit them to our heroku repository.(This is assuming you have an account in heroku. If not please make one, and download Heroku CLI as these are necessary.)
11. After making sure all files are ready and up to be deployed , we use the "git push heroku master" command to push the files on to the server.
12. And this is how our web application is ready and deployed.

SideNote: - The model has been trained on text data which has paragraphs and special characters. Therefore, for an accurate prediction of the Review, the data needs to contain many words. If the review will be small, it will yield lower accuracy.

If you have any more doubts regarding deployment, please refer to this video: https://www.youtube.com/watch?v=Z1RJmh_OqeA

The files in the repository are the files that we create in the flask application building.
The "MyRidge" file is our model that has been serialized and saved in a file so that  we do not have to train the data all the time.




Thank you for your time.
Abhishek Shinde
1001754842
