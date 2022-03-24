## How to train the model
Refer to the `train_model.ipynb` to retrain the model. Pretrained weights are already available in the model folder for the gensim ldamodel

## How to start the api endpoint 
First install the requirements by running the following command
```pip install -r requirements.txt```
Next, you can run the `app.py` file which contains the code for the api endpoint
The endpoint is started on your localhost at port 5000, 

## How to access the endpoint and get the topics for an email
- Using `curl` you can run a similar query as mentioned
```curl -X POST -F 'data=insert the data to predict here' http://127.0.0.1:5000/gettopics```
make sure to change the port number to your flask endpoint , by default flask server runs on port 5000

- You can also use tools like postman to make the POST requests
![postman querry](images/postman.png)
