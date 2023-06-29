

# Topic:
## "Toxic Comment Classification using LSTM: A Deep Learning Approach for Identifying Harmful Language in Online Discourse"

# Problem Statement:

To protect users from being exposed to offensive language on online forums or 
social media sites, companies have started flagging comments and blocking users 
who are found guilty of using unpleasant language. Several machine learning 
models have been developed and deployed to filter out unruly language and 
protect internet users from becoming victims of online harassment and 
cyberbullying. It will help to detect toxicity in users social media accounts. 
The researchers have also found that experiencing cyberbullying increased 
the risk of thoughts of suicide. Victims share Negative feelings that express 
hardship, thoughts of death, and self-harm are widespread on social media. 
Therefore, using social media to detect and identify suicidal ideation will 
help provide proper intervention that will eventually dissuade others from
self-harming.



People now have the ability to openly voice their opinions on a variety of problem
and events through online forums and social media platforms. These internet 
comments occasionally use explicit language that could offend readers. There are 
other categories that comments using explicit language can fall under, including 
Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate. Due to the fear 
of harassment and abuse, many people refrain from speaking their minds and give up 
on exploring other points of view. Companies have started flagging remarks and 
barring individuals who are found guilty of using foul language in order to prevent 
users from being exposed to inappropriate language on internet forums or social 
media platforms. To filter out the foul language and shield internet users from 
experiencing online harassment and cyberbullying, several Machine Learning models 
have been created and put into use.


![image](https://user-images.githubusercontent.com/94919954/220010639-30879251-79d5-495a-b123-fc07d2c96e6e.png)

# Details of tech-stack and tools to be used in project:

    ## 1. Google Colab.
    Google Colab is a free online platform that allows users to write and run Python code in a Jupyter Notebook-like environment, with access to powerful computing   resources and collaboration tools.
    
    ## 2. Numpy.
      NumPy is a Python library that provides support for large, multi-dimensional arrays and matrices, as well as a range of mathematical functions to operate on     them. It is widely used in scientific computing and data analysis
    
    ## 3. Pandas.
    Pandas is a Python library for data manipulation and analysis, providing easy-to-use data structures and tools for handling and cleaning data, as well as         functions for grouping, filtering, and merging data sets
    
    ## 4. Matplotlib.
    Matplotlib is a Python library for data visualization, providing a wide range of customizable plots and charts to display data in a clear and effective manner. It  can be used for creating publication-quality graphics and visualizations
    ## 5. Keras
    a high-level neural networks API written in Python, which can be used to quickly build and train LSTM models
    
    ## 6. TensorFlow
    an open-source software library for dataflow and differentiable programming across a range of tasks, including machine learning and deep neural networks. It       includes a built-in implementation of LSTM
    
    ## 7. HTML/ CSS/ javascript/ React js for Web development
     HTML is the markup language used to create the structure and content of web pages, while CSS is used to style and format the appearance of those pages. JavaScript is used to add interactivity and dynamic behavior to web pages, while React is a popular JavaScript library for building user interfaces, allowing for more       efficient and modular development of complex web applications.
     
    

# Algorithhms ans models choosen:


LSTM (Long Short-Term Memory) and CNN (Convolution Neural Network)
-> are two popular deep learning models used 
in NLP (Natural Language Processing) tasks. In the context of cyberbullying prevention, these models can be used to 
analyze social media posts and comments to identify instances of cyberbullying. “Talos” library since it will help us 
perform hyperparameter tuning as well as model evaluation.


Some of the modules that could be used in such system are:

• Data Collection: Collecting a large corpus of social media posts and comments, including both cyberbullying and 
non-cyberbullying instances, to use as training data for the model.

• Checking for missing values: Verify the downloaded data for any missing values. Using "isnull" on the training and 
test sets of data

• Text Pre-processing: Cleaning and normalizing the text data, converting it into numerical representations that can 
be fed into the deep learning models.

• Text Normalization: -Eliminating spaces between text characters.

-Eliminating Characters That Recur.

- lowering the case of data.
 
- Elimination of Punctuation

-removing any unused spaces between words.

-deleting "n".

-Remove characters that aren't English.


# MODULES AND  PROPOSED WORK

• Lemmatization:It is the procedure of combining a word's several inflected forms into one unit for analysis.
• Stopwords Removal : Inorder to remove stopwords from data, we took the help of the “spacy” library. Removing stopwords 
ensures that more focus is given on those words that define the meaning of the text.


• Tokenization: It is the process of splitting paragraphs and sentences into smaller units that can be more easily assigned meaning.
We need to break down the sentence into unique words. e.g. “I love cats and love dogs” will become 
[“I”, “love”, “cats”, “and”, “dogs”]. Using the “Tokenizer” class from the “Keras” library.


• Word Embeddings: Representing words in a dense numerical vector space, capturing the semantic meaning of words 
and their relationships to each other


• Indexing — We put the words in a dictionary-like structure and give them an index each e.g. {1: “I”,2: “love”,3: “cats”,4: 
“and”,5: “dogs”}.


• Index Representation- We could represent the sequence of words in the comments in the form of an index, and feed 
this chain of index into our deep-learning model. For e.g. [1,2,3,4,2,5].


• Padding: Variable-length sentences are translated into variable-length sequence vectors, and we are unable to feed 
our deep-learning model with vectors of varying lengths. Padding is used to get around this problem. used the 
“pad_sequences” function from the “Keras” library



In order to ensure that the data that will be used by 
our deep-learning models is as clean as possible, our 
goal in this project was to concentrate on data pre-processing and feature engineering. To further leverage 
the potential of Transfer Learning, we choose to employ 
fastText's pre-trained word embedding

# creation of model:

Once data is being cleanded and preprocessed we will use tensorflow to create model.

The acceptable accuracy of model would be greater than 85%.



# Use Cases:

Any mobile or app can integrate this model in their application to protect their users from hate comments and chats.

Any user who wants to check if his or her post is categorised as cyberbully or not can simply check using or website.


We will be creating  a website which will be showing how our model is classifing different  comments,posts and other stuffs is identified as
cyberbully or not.


## Novelty:

Most of the literature survey we go through is only working on english language
but we will try to create a efficient model which will work for hindi as well as 
hinglish language.


## Sub Problem /Gaps Identified:

Multi-lingual Text Classification

Flagging and Blocking Culprits

Tracking and Tracing of Users

One other implementation in our project

Suicide Checking

Video Recommendation to user who text predicted as attempt to suicide.


Data set :

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge



## Literature survey

#Title: Machine learning methods for toxic comment classification: a systematic review
https://www.researchgate.net/publication/349929587_Machine_learning_methods_for_toxic_comment_classification_a_systematic_review

The paper systematically reviewed various machine learning (ML) methods used for toxic comment classification, which is the task of identifying and flagging offensive or inappropriate comments in online discussion forums, social media platforms, and other online platforms. The paper focused on studies published between 2015 and 2020, and analyzed the strengths and weaknesses of different ML techniques used in this context.

The paper identified several common problems tackled by the ML methods, including detecting different types of toxicity (e.g., hate speech, cyberbullying, profanity, etc.), handling imbalanced data, handling multi-lingual comments, and addressing context-dependent interpretations of toxicity. The paper also reviewed various datasets used for toxic comment classification, highlighting their limitations and biases.

The paper identified several problems that remain in this area of research, such as improving the interpretability and explainability of ML models, addressing ethical and privacy concerns, and developing more robust and effective techniques for detecting subtle and nuanced forms of toxicity.

Overall, the paper provides a comprehensive overview of the state-of-the-art ML techniques used for toxic comment classification, and highlights important challenges and research directions in this area

#Title: Imbalanced Text Features for Toxic Comments Classification
https://www.researchgate.net/publication/358140618_Imbalanced_Text_Features_for_Toxic_Comments_Classification

The paper proposes a novel approach for addressing imbalanced datasets in toxic comment classification using imbalanced text features. The authors use text features that are representative of the minority class to augment the data and improve the performance of the classifier.

The proposed approach is evaluated on the benchmark dataset of the SemEval-2019 Task 6 on identifying and categorizing offensive language in social media. The experimental results demonstrate the effectiveness of the approach in improving the performance of the classifier on the minority class.

Overall, the paper presents a promising approach for addressing imbalanced datasets in toxic comment classification using imbalanced text features. However, further research is needed to evaluate the approach on other datasets and to compare it with other methods for addressing imbalanced datasets


#Title: Detection of Suicidal Ideation on Social Media: Multimodal, Relational, and Behavioral Analysis
Link to the paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7381053/    

The paper presents a multimodal approach for detecting suicidal ideation on social media using a combination of linguistic, relational, and behavioral features.

The authors use data from a public social media platform and develop machine learning models to classify posts as either suicidal or non-suicidal. The paper discusses the features used for classification, including linguistic features such as sentiment and topic, relational features such as network centrality and social support, and behavioral features such as posting frequency and time of day.

The experimental results show that the multimodal approach outperforms unimodal approaches that use only linguistic or behavioral features. The paper also discusses the limitations of the study and the ethical considerations of using social media data for suicide prevention.

Overall, the paper presents a promising approach for detecting suicidal ideation on social media using a combination of multimodal features. However, further research is needed to validate the approach on other datasets and to develop effective interventions based on the detected suicidal ideation




#Title: A Comparative Study of Machine Learning Approaches for Toxic Comment Classification
https://www.mdpi.com/2076-3417/10/23/8631

The paper presents a comparative study of different machine learning approaches for toxic comment classification, including logistic regression, decision tree, random forest, support vector machine, and neural network.

The authors evaluate the performance of these approaches on a publicly available dataset and analyze the impact of different features and hyperparameters on the performance of the classifiers. The experimental results show that the neural network outperforms other machine learning approaches in terms of accuracy, precision, recall, and F1 score.

Overall, the paper provides a comprehensive evaluation of different machine learning approaches for toxic comment classification and highlights the advantages and limitations of each approach. The results of the study can guide researchers and practitioners in selecting the most suitable approach for their specific needs.


#Title: A Comparative Analysis on Suicidal Ideation Detection Using NLP, Machine, and Deep Learning
Link to the paper: https://www.researchgate.net/publication/360271430_A_Comparative_Analysis_on_Suicidal_Ideation_Detection_Using_NLP_Machine_and_Deep_Learning

The paper presents a comparative analysis of different natural language processing (NLP), machine learning, and deep learning approaches for detecting suicidal ideation in social media posts.

The authors evaluate the performance of these approaches on a publicly available dataset of suicidal and non-suicidal posts and analyze the impact of different features and hyperparameters on the performance of the classifiers. The experimental results show that the deep learning approach outperforms other machine learning and NLP approaches in terms of accuracy, precision, recall, and F1 score.

Overall, the paper provides a comprehensive evaluation of different approaches for detecting suicidal ideation and highlights the advantages and limitations of each approach. The results of the study can guide researchers and practitioners in selecting the most suitable approach for their specific needs.


##A separate file included in repo which include all literature survey.

Link for research papers:
https://drive.google.com/file/d/1FnQrVgykgqXAJacxEJKKgO_XYUjWTHUD/view?usp=sharing



##LINK TO GOOGLE COLAB
https://colab.research.google.com/drive/1G1cBPB9iOIS8gZJkspi3F0ZSRF5-uTtm?usp=sharing

Data set :

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

The dataset consists of over 150,000 comments labeled with one or more of six categories of toxicity: toxic, severe toxic, obscene, threat, insult, and identity hate. The challenge provided participants with a training dataset for model development and a test dataset for evaluation. The challenge aimed to improve online conversations by identifying and removing toxic comments, thereby promoting a safer and more inclusive online community.


## References

[1]https://www.researchgate.net/publication/360271430_A_Comparative_Analysis_on_Suicidal_Ideation_Detection_Using_NLP_Machine_and_Deep_Learning 

[2]https://www.researchgate.net/publication/358140618_Imbalanced_Text_Features_for_Toxic_Comments_Classification

[3] https://www.mdpi.com/2076-3417/10/23/8631

[4] https://www.hindawi.com/journals/cin/2022/8467349/?utm_source=google&utm_medium=cpc&utm_campaign=HDW_MRKT_GBL_SUB_ADWO_PAI_DYNA_JOUR_X_Partners_Others&gclid=CjwKCAjw8-OhBhB5EiwADyoY1UXfoyQUXm1bur4j1VrVoeGxPzhogH-uHon9oLFIVPT02JLJ-TXHjxoCnCUQAvD_BwE 

[5]https://www.researchgate.net/publication/360271430_A_Comparative_Analysis_on_Suicidal_Ideation_Detection_Using_NLP_Machine_and_Deep_Learning 

[6]  PinkeshBadjatiya, Shashank Gupta, Manish Gupta, and Vasudeva Varma. Deep learning
for hate speech detection in tweets. In Proceedings of the 26th International Conference on
World Wide Web Com anion, a es 759-760, 2017  
https://arxiv.org/abs/1706.00188

[7] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7381053/
Detection of Suicidal Ideation on Social Media: Multimodal, Relational, and Behavioral Analysis



