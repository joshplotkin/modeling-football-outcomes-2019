### Work completed
* 00_source_data
  * sourced first iteration of data from around 10 sources. (scrubbed website names)
  * identified potential future sources.
* 01_create_hive_data_model
  * QA'ed, cleaned, enriched, and did initial exploration of the data.
  * created a star schema and wrote to Hive tables, to be queried by Spark.
    * Spark isn't necessary for this data but the DataFrames API is great, and it makes the code scalable. the same code can run on a Spark cluster to handle more data.
* 02_exploration & 03_labels
  * initial data exploration and visualization. 
  * prepared labels and wrote to Hive tables.
* model_pipeline/ and 04_model_pipeline_dev.
  * developed model train/score/evaluate/visualize pipeline with model repository, for rapid feedback when engineering features and improving the model.
    * python and bash execution files in model_pipeline directory.
    * execution driven through 2 JSON configuration files. place them in directory models/{MODEL_ID} and run: model_pipeline/model_pipeline.sh {MODEL_ID}
    * core ideas developed with Pivotal teammate, Tim Kopp. learned many great ideas from Tim, like incorporating model choice into model config using importlib, the directory structure, using directories as a model repository, notebook to generate JSON, among many others. code is my own.
* model_pipeline/generate_configs_execute.ipynb
  * notebook to generate JSON configuration files.
  * thoroughly tested pipeline.
* 05_feature engineering
  * feature engineering performed across a series of notebooks.
* next steps
  * continued feature engineering.
  * model selection.
  * model evaluation and "where did we miss?"
  * model explainability and feature contribution.
* backlog
  * model pipeline test cases.
  * regression model support.
  * continuous stratified sampling columns (using binning).
  * Spark ML support for pipeline.
  * additional data (more below).
  * pipeline to ingest incremental data.

### ER Diagram showing data model 
* automated using __eralchemy__
* code to do this is in the ER diagram notebook

![E-R Diagram for Database][erd]

[erd]: img/er_diagram.png

### Data
* don't have passing vs. running offense or defense stats
* have injury data but have not incorporated. could be extremely useful.
* plan is to eventually add advanced stats
   * profootballfocus
   * sports info solutions
   * sharp football
   * NFL nextgen
   * historical prop bet lines?
   * pregame.com
   * would be nice to get schemes and dig into specific matchups, e.g. WR vs CB or O vs. D schemes
* how to encode team changes in the off-season? e.g. FA acquisitions

### Open Questions
* for labels, each row is a game
  * which team should the spread be based off of?
  * it could be home team, favorite, random, or something else
  * home team team adds bias. it might be better to encode Home/Visitor as a feature
  * favorite adds bias but perhaps more tolerable
* how to handle early in the season vs. later? maybe start out building model for weeks 8-17?
* how to handle playoffs vs. regular season?
* how to figure out how to incorporate injuries
* playoff scenarios: sometimes one team is motivated while another is not (or is tanking)
  * week 17 can be tricky 