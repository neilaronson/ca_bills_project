# Predicting Bill Success in the California Legislature

Each session, the California Legislature considers on average about 5,400 bills, 36% of which eventually become law. These bills can affect everyday life for 39 million Californians, from the state of our roads to the minimum wage. The goal of this project is to figure out which bills are mostly likely to pass so that citizens, NGOs and other interested parties can prioritize which bills they should focus on. I also wanted to see if there were patterns in the data that could reveal insights about the legislative process.

## Overview

I used a [MySQL database made available by the Legislature](http://downloads.leginfo.legislature.ca.gov/) that contains three types of data about bills: metadata (such as when a bill was introduced), author data (such as the party of the author), and the text of the bill. I used all bills from 2001-14 as a training set and all bills from the 2015-2016 session as a test set. My goal was to predict whether a bill will pass both houses of the Legislature given what we know about a bill when it is first introduced.

I was surprised to find that many factors I thought would be important did not seem to make a difference, particularly the seniority of the legislators who introduced the bill and their past success rate. This seems likely to be because of relatively strict term limits in California. The three main factors that did move the needle were party, taxes and latent topics. The first two were relatively obvious. Democrats have controlled the California Legislature continuously since 1997, and bills sponsored by Republicans have a much lower chance of passing. Taxes are also hard to pass, particularly given that they require a 2/3 vote of the Legislature. But the most interesting finding was how latent topics extracted from the bill text contributed toward bills' chances.

![Percent of passed by party](https://github.com/neilaronson/ca_bills_project/blob/master/graphs/party_05-24-17-16-15.png)

Latent topics are topics extracted algorithmically from text that arise from words or phrases occurring frequently together. I used TF-IDF to calculate term frequency and Non-negative Matrix Factorization to extract the latent topics. Experimentation showed that using 100 latent topics provided the best performance in my model. Some examples of latent topics detected in bill text were tax credits, veterans' benefits, and school district governance. But the latent topics that had the biggest impact on a bills' chances were surprising.

![Latent topics diagram](https://github.com/neilaronson/ca_bills_project/blob/master/graphs/latent_topics.png)

Number one was what I call the "Memoranda of Understanding" topic, which refers to agreements already negotiated between the state and its workers that the Legislature is legally required to approve. The Legislature usually rubber-stamps these bills. The second most important topic was what I call the "Good Intentions" topic. Sometimes legislators will introduce a one-word placeholder bill that states simply, "It is the intent of the Legislature to enact legislation relating to [insert topic here]." Although legislators come back and later fill these out, it turns out this is a bad way to start a bill; these bills are less likely to eventually pass.

![Topic dependency graph](https://github.com/neilaronson/ca_bills_project/blob/master/graphs/topic_dependency.png)

I found that the latent topic features contributed to better recall, while the party and bill metadata features provided for better precision. In my final model, I used a Random Forest ensemble with a max depth of 8 which allowed for interactions between topic and party. The baseline I compared my model against was the 44% overall pass rate of bills, which yields an F1 score of .44 and an accuracy of .51. My Random Forest model was able to achieve an F1 score of .71 and an accuracy of .62.

![Bill model results table](https://github.com/neilaronson/ca_bills_project/blob/master/graphs/results.png)

There are numerous next steps I'd like to take to improve the model. The most important source of variation that I believe remains unaccounted for in my model is the level of public interest in a bill's topic (or in the bill itself). I would like to add in proxies for this interest level by scraping California newspaper articles or search engine data. I would also like to be able to build a multistage model that can make better predictions as more is known about a bill, perhaps with a generalized linear model that can deal with non-independent observations arising from multiple versions of the same bill. Ultimately I hope to make this model available to the public to track ongoing bills in the California Legislature so they can know what bills are most likely to pass.
