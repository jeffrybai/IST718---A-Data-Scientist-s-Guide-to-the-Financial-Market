# Financial Forensics

Machine Learning Analysis of SEC 10-K Disclosures

Jeffry Bai

jbai06@syr.edu

December, 2023

IST 707 Applied Machine Learning Final Project

Introduction

Enron Corporation was once a titan of industry who was a major player in electricity, natural gas, communications, and pulp and paper. The company was a darling of Wall Street and Fortune named it “America’s Most Innovative Company” for six consecutive years. During 2000, Enron’s stock was trading at &#36;90.56 and claimed &#36;101 billion worth of revenue. Less than a year later, Enron filed for Chapter 11 bankruptcy with its stock closing at &#36;0.26. Turns out,Enron had been inflating its income by around &#36;586 millions dollars since 1997 and hid financial losses using mark-to-market accounting practices. Enron would measure the value of a security based on its current market value instead of book value, claiming the projected profit of an asset on its books. When the revenue of said asset came under the projection, the company would then transfer the asset to an off-the-books corporation instead of recognizing the loss. These accounting practices are fraudulent and unsustainable which caused the implosion of Enron. The collapse of Enron resulted in multiple criminal charges filed against its executives, the dissolution of a major accounting firm Arthut Anderson, and the introduction of bipartisan Sarbanes-Oxley Act of 2002 in effort to prevent future fraud and mismanagement. 

While the Sarbanes-Oxley Act (SOX) did put additional guardrails in place, the controls did not stop companies from committing financial fraud and making misleading statements. Companies such as American International Group, General Electric, Under Armour all later found to have committed financial misconduct post-SOX. Other companies such as Wells Fargo and Nikola, while not committing outright financial crimes, were found to have made misleading statements in effort to boost investor confidence. These statements were later found to be materially false, which led to investors' loss. One of the governing bodies overseeing companies’ compliance to SOX is the U.S. Securities and Exchange Commission (SEC). SEC requires publicly traded companies to file documents that disclose information about their financial condition, operations, and ownership. These documents, known as SEC filings, are not only an important tool for regulatory enforcement but also a great resource for investors to be informed. 

These documents, while crucial for regulatory enforcement and investor information, are often dense and complex, requiring significant effort, domain knowledge, and expertise to dissect. With advancements in Machine Learning (ML) and Natural Language Processing (NLP), new avenues have emerged for in-depth analysis of these intricate documents. This project focuses on applying various ML algorithms to a compilation of 10-K annual filings,aiming to computationally identify indicators of fraud, thereby paving the way for more nuanced and technologically advanced approaches in financial analysis and regulatory oversight. 

# Analysis and Models

# Data Exploration

While SEC requirements entails a variety of filings for public companies, the file that is the focus for this analysis is 10-K. The 10-K report is a comprehensive annual filing mandated by the SEC, it provides a detailed overview of a publicly traded company's financial performance, operations, risk factors, and management's analysis. It is an information dense document that contains crucial information that is ideal for this analysis. The raw text files were downloaded from SEC Electronic Data Gathering, Analysis, and Retrieval (EDGAR) library via python. A total of 445 10-K from 24 unique companies were analyzed and manually labeled. Based on research of publicly available information, filings were labeled either as ‘financial’ if the company was found to have committed financial fraud, ‘misled’ if the company was found to have made materially false statements that misled investors, or ‘amendment’ if the 10-K was later amended. For binary analysis, an additional column is added to identify if companies had included potentially misinformation on 10-K. This column is True if the 10-K was labeled ‘financial’, ‘misled’, or ‘amendment’ and False if otherwise. Out of the 445 files analyzed, 95 were labeled to have contained misinformation. Out of the 95 files, 49 were amended, 30 represented the years that said company committed financial fraud, and 16 represented the years the company misled investors. 


![](https://web-api.textin.com/ocr_image/external/57f1f4956cc87b56.jpg)


![](https://web-api.textin.com/ocr_image/external/4cbf775b1f287d53.jpg)

Fig 1. Breakdown of misinformation labels on 10-K filings. Chart on the Left shows the total number of filings that contain misinformation; the chart on the Right shows the breakdown of the type of misinformation. 

Since the raw files contains a markup language that varies over the years, several clean up steps were taken; these steps include: 

1.​ Removing all $<DOCUMENT>$ blocks of the $<TYPE>$ GRAPHIC, ZIP, EXCEL, PDF, JSON, XML, EX-. These blocks are exhibits and supplemental information that is not particularly useful in this analysis. 

2.​ Removing a $ll<script>,$  &lt;Logs&gt; blocks. These are blocks that contain javascripts or error logs for the SEC web application to function, having no material impact to this analysis. 

3.​ Removing text enclosed in angle $(<>)$  and square brackets ([ ]). This removes all markup language and unnecessary details. 

4.​ Remove all URLs and any strings that end in .htm, .xsd, .xml, .jpg, .txt, .sgml.

5.​ Remove all symbols and numbers. Numbers would only be useful when performing a financial analysis. It is unfair to compare the finances from different companies. 

6.​ Finally 4 different final text files were generated for various analyses.

a.​ A cleaned text file ran through steps 1 through 5.

b.​ An english only file which effectively removed certain proper nouns.

c.​ A file with the words replaced with its respective parts of speech (POS).

d.​ A file with the english words lemmatized.

# Models and Method

For this project, the following algorithms were used.

1.​ Naive Bayes is a family of classification algorithms based on applying Bayes' theorem with the assumption of independence between features. Variations includes: 

a.​ Gaussian Naive Bayes assumes features follow a normal distribution and is particularly suited for continuous data. 

b.​ Multinomial Naive Bayes is ideal for classification with discrete features.

c.​ Bernoulli Naive Bayes is effective for models with binary inputs.

2.​ Support Vector Machines (SVM) a set of supervised learning methods particularly effective in high dimensional spaces. For this analysis, the kernels linear, polynomial, radial basis function, and sigmoid are evaluated. 

3.​ Decision Tree Classifier models decisions and their possible consequences.

4.​ Random Forest Classifier is an ensemble learning method that operates by constructing a multitude of decision trees for robust classification. 

5.​ K-Nearest Neighbors Classifier (KNN) is an instance-based learning method used for classification that predicts the label based on the majority label of its closest neighbors. 

The cleaned, english only, part of speech tag, and lemmatized text files are converted into term frequency inverse document frequency (TF-IDF) and fed into a pipeline of standard scalar then the respective models to compare which combination had the best performance. The best performing models are then isolated for fine tuning. 

# Results

Since the dataset is heavily skewed towards files without misinformation, the no information rate (NIR) is assumed to be the majority class of $350/445$  at $78.7\%$ . When comparing model performance of category with nominal versus binary label, it is clear that 4 nominal labels performed better; in fact, only 2 instances yielded a result that was marginally higher than NIR rate. This is rather unexpected since nominal label segments data further,meaning less data for training. Observation could suggest that nominal labels separates the data better than binary grouping. Most models did not perform well and failed to beat NIR rate, with exception to KNN and RBF SVM with nominal label. One observation is that both KNN and RBF utilize distance measures in their algorithm, albeit differently. This suggests that there is a proximity relationship between filings that contains misinformation. POS as a preprocessing method only performed marginally better in SVMs, while this could be within an experimental error, it warrants further experimentation. 


| Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ | Nominal Label Binary Label <br>$NIR=78.7\%$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Cleaned  | English only  | Lemma  | POS  | Cleaned  | English only  | Lemma  | POS  |
| KNN  | $81.10\%$  | 79.30%  | 79.76%  | $71.65\%$  | $75.74\%$  | 76.22%  | 76.44%  | 74.64%  |
| GaussianNB  | $63.68\%$  | 63.46%  | 63.02%  | $20.03\%$  | $62.40\%$  | 61.99%  | 62.21%  | 30.37%  |
| MultinomialNB  | $34.03\%$  | 49.57%  | 49.79%  | $74.82\%$  | 29.06%  | 45.16%  | 45.16%  | $70.99\%$  |
| BernoulliNB  | $76.17\%$  | 51.93%  | 52.37%  | $75.28\%$  | $73.56\%$  | 64.57%  | 63.68%  | $75.06\%$  |
| GradientBoosting  | $62.85\%$  | 62.02%  | 64.29%  | $70.79\%$  | $56.88\%$  | 60.21%  | $57.72\%$  | $65.86\%$  |
| Linear SVC  | $54.50\%$  | 61.48%  | 61.26%  | $77.58\%$  | 49.44%  | 46.76%  | 46.98%  | 66.13%  |
| Polynomial SVC  | $77.54\%$  | 71.41%  | 71.86%  | 79.57%  | 77.56%  | 66.09%  | $66.55\%$  | $69.95\%$  |
| RBF SVC  | $81.33\%$  | 82.23%  | 82.01%  | $78.66\%$  | $76.44\%$  | $75.30\%$  | $74.85\%$  | $75.09\%$  |
| Sigmoid SVC  | $76.16\%$  | $76.63\%$  | $76.40\%$  | $78.66\%$  | $72.43\%$  | $63.48\%$  | $63.48\%$  | $78.66\%$  |
| DecisionTree  | $58.87\%$  | $53.17\%$  | $55.26\%$  | $53.28\%$  | $51.24\%$  | $50.27\%$  | $51.80\%$  | $54.39\%$  |
| RandomForest  | $76.61\%$  | $75.49\%$  | $75.03\%$  | $77.10\%$  | $71.29\%$  | $67.90\%$  | $70.83\%$  | $70.58\%$  |


Fig 2. Table of mean accuracy for 10 fold cross validation with 4 category nominal labels on the left and binary 

category on the right.

Looking at the top performing model, a question came to mind: would more data yield better results? Taking the same dataset, we can arbitrarily increase the training data set by increasing the number of folds in the cross validation. As presented in Fig. 3, it supports the conclusion that more data improves model performance. Between 10-fold and 50-fold, the number of training data increased by about 20, with an improved mean accuracy of just over $2\%$ . Between 50-fold and 100-fold with an increase of training data of about 20, the improved mean accuracy was less than $0.5\%$ . This suggests that as model performance increases, there is a diminished return with each additional training data. 


| English only  | 10 Fold  | 50 Fold  | 100 Fold  |
| --- | --- | --- | --- |
| RBF SVC  | 82.23%  | 84.31%  | 84.75%  |


Fig 3. Mean accuracy performance comparison of 10-fold, 50-fold, and 100-fold cross validation. 

Accuracy score of 82.23% at default parameters is hardly definitive. Taking the top performing models and iterating through a set of parameters, Figs. 4 through 6 shows the top mean accuracy with 100-fold cross validation and respective parameters. The accuracy was able to be improved to above 85% with SVM with the parameters {'C': 0.1, 'coef0': 10, 'degree': 5, 'gamma': 1.0, 'kernel': 'poly'}. Since a degree of 5 was the highest parameter iterated through, there is potential that model performance could improve at higher polynomial degrees. 


| KNN  | Cleaned  | English only  | Lemma  |
| --- | --- | --- | --- |
| KNN  | 85.00%  | 84.35%  | 84.35%  |


Fig 4. Top performing KNN models with parameters of {'metric': 'euclidean', 'n_neighbors': 13, 'weights': 'distance'}

with the mean accuracy of 100-fold cross validation


|  <br>RBF SVC  | Cleaned  | English only  | Lemma  |
| --- | --- | --- | --- |
|  <br>RBF SVC  | 84.05%  | 84.05%  | 83.85%  |


Fig 5. Top performing KNN models with parameters of {'C': 1, 'gamma': 10, 'kernel': 'rbf'} for cleaned and english preprocessing and {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'} for lemma with the mean accuracy of 100-fold cross 

validation


|   | POS  |
| --- | --- |
| Polynomial SVC  | 85.35%  |
| RBF SVC  | 83.45%  |
| Sigmoid SVC  | 80.00%  |


Fig 6. Top performing SVM with parameters from top to bottom of {'C': 0.1, 'coef0': 10, 'degree': 5, 'gamma': 1.0, 'kernel': 'poly'}, {'C': 1000, 'gamma': 1, 'kernel': 'rbf'}, & {'C': 0.1, 'coef0': 0, 'gamma': 'scale', 'kernel': 'sigmoid'}

with the mean accuracy of 100-fold cross validation.

# Conclusion

While an accuracy of 85% is not a bad result, suggesting that there is application for Machine Learning in financial analysis of 10-K filings, it is important to put context around the result for interpretation. In the dataset, there are files with misrepresented financials or misleading information that are only associated with certain companies. Enron, for instance, only had categorization of having committed financial fraud. The relatively high accuracy with cleaned files could be attributed to the models ability to distinguish one company’s filing from another. Since the businesses of these respective companies are unique to a degree, similar explanations can be provided for english only and lemmatized preprocessing methods. 

One area that provides optimism is the result from parts of speech analysis. Since parts of speech removes all context of a document, the machine learning algorithm cannot be trained on vocabulary that are unique to a company’s business, industry, or other identifiable features. The return suggests that there are distinguishable lexical differences between filings with and without misinformation. One possible explanation is that there are differences in lexicon between companies, which is the underlying feature that the model was identifying. While inconclusive, it warrants further investigation. 

Study also suggests that more data would improve model performance, although with a diminished return with each additional training data. A path forward would be to increase the dataset significantly, optimize prediction with an ensemble method, and include financial analysis to provide more contextual information. Even though more work is involved to get more conclusive results, this is a demonstration of the power of Machine Learning in the application of a very complex domain. While the results are encouraging, they also highlight the need for further, more granular analysis to truly understand the capabilities and limitations of Machine Learning in this context. Such an investigation would be invaluable in refining the model's application, ensuring it reliably identifies misinformation in financial filings across diverse company profiles and industry sectors. 

