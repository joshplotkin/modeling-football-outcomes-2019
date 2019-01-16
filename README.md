### Work completed
* public00_source_wrangle_data.ipynb
  * sourced first iteration of data from around 10 sources. identified potential future sources. (scrubbed website names) 
* public01_write_tables.ipynb
  * QA'ed, cleaned, enriched, and did initial exploration of the data.
  * created a star schema and wrote to Hive tables, to be queried by Spark.
    * Spark isn't necessary for this data but the DataFrames API is great, and it makes the code scalable. the same code can run on a Spark cluster to handle more data.
* public02_explore_scores_lines.ipynb & public03_form_visualize_labels.ipynb
  * initial data exploration and visualization. 
  * prepared labels and wrote to Hive tables.
* src/ & public04_train_score_evaluate_pipeline.ipynb.
  * developed model train/score/evaluate/visualize pipeline with model repository, for rapid feedback when engineering features and improving the model.
    * python and bash execution files in src directory.
    * model execution driven through 2 JSON configuration files. place them in directory models/{MODEL_ID} and run.
```bash
src/model_pipeline.sh {MODEL_ID}
```
    * core ideas developed with Pivotal teammate, Tim Kopp. learned many great ideas from Tim, like incorporating model choice into model config using importlib, the directory structure, using directories as a model repository, among many others.
* public05_generate_configs_execute.ipynb
  * notebook to generate JSON configuration files.
  * thoroughly tested pipeline.
* next steps
  * model pipeline test cases.
  * feature engineering and model development.
* backlog
  * regression model support.

### ER Diagram showing data model 
* automated using __eralchemy__
* code to do this is in the ER diagram notebook

![E-R Diagram for Database][erd]

[erd]: er_diagram.png

  * don't have passing vs. running offense or defense stats
  * hope is to eventually add advanced stats
     * profootballfocus
     * sports info solutions
     * sharp football
     * NFL nextgen
     * would be nice to get schemes and dig into specific matchups, e.g. WR vs CB or O vs. D schemes
  * how to encode team changes in the off-season, e.g. FA acquisitions

### High level plan; filling in with ideas as they come up
* source/scrape/wrangle data
* clean/combine data and build a data model
* identify labels
  * label could be classification: spread, straight up winner, over/under
  * could be regression (score/margin of victory)
  * could be classification (did/didn't cover)
  * how to decide in which direction label goes (e.g. who is 0 or 1 in classification...  is it home team? favorite? random?)
  * explore/visualize labels
* explore/visualize data
  * look at correlations to label
* start with simple features
* build CV/evaluation framework
 * for spreads, use score as confidence metric. type II rate at each score bin
 * stratify as needed
* model evaluation
  * this isn't a model that will have a 0.99 AUC
  * Vegas Super Context winners have 60-65% accuracy, on only 5 games.
    * this would be an example of an evaluation technique. Take top 5 highest scores and look at accuracy.
* train an early model using CV framework
* engineer features
  * raw fields
    * consecutive road games
    * using prop bet lines? (historical data will be hard to find)
    * team's record vs. coach
    * previous team(s) faced (ranks as proxy for teams)
    * some proxy for home field advantage: wins above expected?
    * QB recent matchups with team
    * home team (binary)
    * recent record against similar opponents
    * O/D/ST/overall rankings
    * divisional matchup indicator
    * travel distance from last game (decayed by time?)
    * time since last game
    * week of season
    * record/finish last season
    * playoff situation (is a win needed?)
    * team fanbase size
  * player rankings, e.g. QB?
   * Mon/Thu/primetime: 
  * Travel: euclidian distances between cities?
  * how many days ago was last game? covers bye week too
  * timezone change
  * past matchups between the teams
  * team records and rankings
  * team average/median age?
  * current injuries
   * could be number of snaps per healthy game at given position(s)
* rinse and repeat:
  * build more features
  * feature selection
  * bring in more diverse data as needed
    * e.g. injury info could be important but probably not needed in first iteration
  * model selection/evaluation
* model explainability: understanding why the model is making certain picks
* pipeline to ingest new data

### open questions
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

### future data assets
  * have injury data but have not incorporated. could be extremely useful.
  * don't have passing vs. running offense or defense stats.
  * hope is to eventually add advanced stats.
     * profootballfocus
     * footballoutsiders
     * sports info solutions
     * sharp football
     * NFL nextgen
     * would be nice to get schemes and dig into specific matchups, e.g. WR vs CB or O vs. D schemes
  * how to encode team changes in the off-season, e.g. FA acquisitions
