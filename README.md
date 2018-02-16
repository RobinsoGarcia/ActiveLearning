THIS README FILE IS A WORK IN PROGRESS! I suggest checking the pdf file and the bibliographic references for a better understanding of the code.

[1] class active():
    def __init__(self,X=[],y=[],X_=[],y_=[],method='random',density_method='none',sparse=False):

    X,y: Train data and labels
    X_,y_: Test data and labels
    method: type of active learning approach (see final_paper.pdf)
        eg: 'random', 'least_confident', 'margin' or 'entropy'

    density_method: 'none', 'euclidean' or 'cosine' (cosine similarity)

    sparse: True or False, indicate if the input data is sparse or not.

[2] relevant attributes:

    self.clf = LogisticRegression(penalty='l2',C=1):
      it can take the form of any classifier as long as the methods fit and predict_proba are available.

    self.sample_size=10
      Number of samples to select from the pool at each query

    self.min_pool=20
      minimum number of datapoints in the pool (starting point)

    self.set_pool()
      initialize the pool

    self.update()
      update the pool

    self.qbc method
      query method used by the committee 'least_confident', 'margin' or 'entropy'

    self.qbc_sampling
      percentage of the data sampled at each iteration (bootstraping)

    self.n_splits=3
      number of commitees formed using the boostraped data.


[3] Some possible scenarios *

      scenario_data = {'uncertainty':{0:['least_confident','none','none'],
      1:['margin','none','none'],
      2:['entropy','none','none'],
      3:['least_confident','euclidean','none'],
      4:['margin','euclidean','none'],
      5:['entropy','euclidean','none']}}
      6:{'boosting':{0:['boosting','euclidean','margin']},
      'bagging':{0:['bagging','euclidean','margin']}}

[4] Example:
  result = {}
      for j in np.arange(n):
          obj = active(X,y,X_,y_,method=case[0],density_method=case[1])
          obj.qbc_sampling=0.4
          obj.n_splits=1000
          obj.min_pool = min_pool
          obj.sample_size = sample_size
          obj.set_pool()
          obj.set_density()
          obj.iter=iter_
          obj.qbc_method = qbc_method
          obj.fit()
          result[j*sample_size]=obj.accuracy_bin


The codes were used with a structure of folders that pre-existed in my computer. Thus the main script will ask for folders that won't exist on a third party computer. I haven't organized an implementation that creates all folder automatically, but I'm attaching the files to illustrate the ideas.

The main archive is active.py which is a class that does all active learning work. This is a work in progress but for the purpose of the final project, it has all implementations ready to go. Since providing these files werent required I haven't added explanations. However, looking at the paper's diagram, the main file, and the active class should give a good idea of whats happening in the background.

Again there are implementations on this file that goes beyond the scope of the project. For instance, there is a function called clustering, which is a preliminary test on using data structuring, bagging and stochastic method to select new queries (please disregard, though it is working).
