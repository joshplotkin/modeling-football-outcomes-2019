### Current status
* currently exploring the outcomes and lines, and generating lots of plots
  * next step is working to generate a __labels__ table, with various labels
    * binary did_cover_spread
    * actual margin
    * will have to choose in which "direction", e.g. do I always keep the line relative to favorite?
    * over/under
  * next step after will be to build the model training/scoring/evaluation scaffolding
* sourced and enriched most of the desired data. some remain:
  * sourced injury data but didn't clean/incorporate
    * incorporating will be a challenge
    * snap counts could help find important players
    * it will be critical to encode things like the starting QB being out
    * __note: due to injuries and not knowing what information was available when the line opened, might need to use only the closing line and favorite__
  * sourced play-by-play data but didn't clean/incorporate
  * don't have passing vs. running offense or defense stats
  * hope is to eventually add advanced stats
     * profootballfocus
     * sports info solutions
     * sharp football
     * NFL nextgen
     * would be nice to get schemes and dig into specific matchups, e.g. WR vs CB or O vs. D schemes
  * how to encode team changes in the off-season, e.g. FA acquisitions
* cleaned data and settled on data model in Hive via Spark
  * games_denorm (all fields in denormalized schema)
  * game (fact table)
  * game_metadata
  * game_outcome
  * game_line
  * team_season
  * coach
  * team
  * stadium

### ER Diagram showing resulting schema
* automated using __eralchemy__
* code to do this is in the ER diagram notebook

![E-R Diagram for Database][erd]
[erd]: er_diagram.png


### High level plan; filling in with ideas as they come up
* source/scrape/wrangle data
* clean/combine data and build a data model
* identify labels
  * label could be classification: spread, straight up winner, over/under
  * could be regression (score/margin of victory)
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
    * home team (binary)
    * O/D/ST/overall rankings
    * divisional matchup indicator
    * travel distance from last game (decayed by time?)
    * time since last game
    * week of season
    * record/finish last season
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

