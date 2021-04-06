# Riiid Answer Correctness Prediction : LGBM + SAKT

My training & submission **notebook** and **write-up** for the Kaggle competition [Riiid Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction/overview).


---

**Final score** : 0.793 (public) / **0.794 (private) (rank: 143 - Solo silver medal (top 5%))**
- Rank-1 score : 0.820
- Silver-to-gold score : 0.815
- Bronze-to-silver score : 0.792
- Lower Bronze score : 0.788

---

### About the competition

- Large tabular dataset (>5GB) of users' interactions with an app for learning english (TOEIC test). A few features : user_id, content_id, tags, prior_question_had_explanation, prior_question_elapsed_time, timestamp, task_container_id.
- About knowledge tracing (KT). Target : answered_correctly (probability for the user to correctly answer to the current question). 
- Key aspects of the competition :
	- Machine learning : small number of features => room for heavy FE
	- Software Engineering : large dataset, Kaggle environment limited to 16GB RAM, submission time < 9 hours, submit predictions through batches delivered by an API, heavy state variable to store / update on-the-fly.
- Top solutions involves :
	- LGB (or [CatBoost](https://arxiv.org/abs/1706.09516) + Transformer-based model ([SAINT+](https://arxiv.org/pdf/2010.12042.pdf), sota model on KT from the hosts)
	- Pure Transformer-based solutions
	- A couple of pure LGB-based solutions (competitive, but less efficient overall)
	- A couple of exotic solutions relying on LSTM-like models (GRU)



### Solution overview


<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/riiid_lgbm_sakt/master/assets/riiid_ensemble_pipeline.jpg">
</p>


My final model is a [LGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) + [SAKT](https://arxiv.org/abs/1907.06837) ensemble that heavily rely on feature engineering. 

Most of my choices were conditionned by the limited computational resources I had (16GB CPU RAM).
Pandas data manipulation functions applied on such a large dataset almost always lead to a RAM overflow (under the Kaggle environment limitations). I decided to rely on loops + numpy functions for feature engineering :
- No time wasted for optimizing preprocessing functions to make them fit the RAM ; writing function for FE was easy and fast, and I only had to compute them once.
- Significantly limiting implementation errors : I used the exact same sequence of instruction for both FE (offline) and inference loop data preprocessing (online).
This approach made the workflow simple and systematic, allowing fast debugging. 



### Validation strategy 
- Used a [CV approach shared be tito](https://www.kaggle.com/its7171/cv-strategy), another Kaggle competitor. The key idea is to distribute users in the dataset so that : 1) there is no time leakage (i.e. if t1>t2, row_index_1 > row_index_2), and 2) users are uniformly distributed w.r.t. the time they spent on the app. This strategy efficiently mimic the test dataset behavior, in which there are known and experimented users, as well as newcomers.

Actually, the evaluation datasets were so large that such a simple cv leads to a perfect CV to LB correlation. 
I relied on CV + LB scores all along the competition for making model design choices. I switched to another CV split seed a couple of time during the competition to make sure that the model was not overfitting.
My final submission had a slightly better LB score in the private dataset than in the public one.

### Setup
As I couldn’t rely on a good local setup, I decided to work on the Kaggle environment.

PROS: No additional time required to make the inference pipeline work under Kaggle env, which was required for submitting.
In this competition, inference must be done into the Kaggle environment to do submissions through an API which sequentially gives batches of test data. Many competitors who worked locally were not able to properly establish their inference pipeline in the Kaggle environment (bugs, limited RAM, heavy offline preprocessed data to upload).

CONS: Limited RAM from training (16gb cpu or 13gb cpu + 16gb gpu)


### Feature Engineering :

My final LGB model relies on 53 features, selected through the following workflow :

New set of features → Evaluation → Feature importance (SHAP) → Discard less important features → ...

Question-based : 
- Avg/std/median score per question, 5-bins split by difficulty level : Was the question difficult ?
- Lag time : How long since the last question ?
- Current part


User-based : 
- Avg score/ question count / sum score from different perspectives (in general / per difficulty level / per part)
- Current / previous right answer streak
- Avg score : last 25 questions, 25 to 50, 50 to 75
- Lag time (last 3 intervals), avg lag time, time since last right / wrong answer
- Time spent on the current question (approximated)  
- Current session time, number of tackled questions
- Avg time per session
- Avg question explanation : Did the user saw many explanation of the tackled questions ?
- Current bundle id : Is the user a newcomer or a veteran ?
- Last bundle score stats
- Current question difficulty level
- Is it the user’s first attempt on the current question ?
- How much unique questions did the user faced in the past ?
- Clustering from the first interactions (first questions are usually similar between users, and a specific sequence of right/wrong answers leads to similar questions & scores)

Ratios : 
- Did the user spent more or less time than other users on a specific question ?
- How long since his/her last tackled question compared to others ? (high lag time could mean a break, or some lectures)
- How much time of practice does the user spent to reach its current performances ? (avg score divided by time spent on the app)

But also :
- Avg score per tags, stats on tag frequency (min/max/mean/std)
- Lecture count per user


Notes :
- I’ve tried > 160 different features, including longer user history score, day/week/month based features, more bundle & lecture based features). Discarding most of them helped generalization.
- While the workflow was stable, it must have been faster with SQLite based approach – using same feature processing function for both training & inference.
- In particular, my attempts to use word2vec and LDA based features did not improve the model. Winning solutions which used those kind of features relied on (user x questions) and (user x tags) large matrices, projecting them into ~5 features vectors. This is definitively something I will experiment in the future.


### LGBM parameters

I added L1 & L2 regularization to limit overfitting. I set subsampling to 50% of rows and 70% of features to deal with RAM limit exceeding.

### SAKT (attention-based) model

SAKT original paper : [link](https://arxiv.org/abs/1907.06837)

This part of my solution relied on a [code shared on Kaggle](https://www.kaggle.com/wangsg/a-self-attentive-model-for-knowledge-tracing) during the competition, making some improvements : 

- Train/Valid user interaction sequences split with respect to timestamp, making sure to simulate private-LB data conditions : validation sequences follow training ones, some users are in both training/valid datasets, and new users are regularly added to the test data. 
- Set max_seq to 180 (maximum length for user interaction sequences) was a good bias/variance compromise, and lead to better CV/LB.

Note : Adding other features to the sequence (part, lag time) deteriorated both the CV & LB score. Some other competitors successfully used those features on transformer-based models (like SAINT+) which can capture more complex patterns in the data.

- Training setup : 10 epochs with lr=e-2, 5 epochs of refining with lr=e-3. Batch size of 64, 10% dropout. Min sequence size to 4, embedded size of 128. Trained on NVIDIA TESLA P100 GPU provided by Kaggle.

### Ensemble

My final solution is a simple weighted ensemble : 70% LGBM, 30% SAKT

Stacking didn’t work well for me. More than two models ensembling too. 

My LGBM alone got 0.789 LB, complementarity with the DL model made a significant improvement.


### Key to get higher in the LB

Most of the winning solutions are purely transformer-based. 

The large dataset, as well as the time-based inference API made those model outperform LGB ones on this task.
Some winning solutions from similar Kaggle competitions were somehow misleading ; many of them heavily relied on LGBM and massive Feature engineering to reach the top of the leaderboard. But those competitions were a little biased compared to Riiid, in a sense that there were no such thing as time-based inference API that prevent competitors to use some forms of time leakage (input features for later interaction that could help to predict target for earlier interactions). In such a setup, decision-trees based models were very efficient, and were over represented at the top of leaderboards.

Some competitors have reported a single-LGB model with 0.801 private-LB. Those solutions relied on a full history for each user, which is heavy RAM consuming for both training & inference. SQLite approaches allowed them to fit those constraints; it would be a great solution to consider in the future when dealing with large dataset in a computationally limited environment.


Deploy a decision-trees trained model using [treelite](https://github.com/dmlc/treelite) to optimize inference speed would have been a great solution to earn time, which could have been allowed to computationally greedy state-variable updates and feature processing.


### Feature importance

<p align="center">
  <img src="https://raw.githubusercontent.com/Johann-Huber/riiid_lgbm_sakt/master/assets/lgbm_feature_importance_barplot.png">
</p>


