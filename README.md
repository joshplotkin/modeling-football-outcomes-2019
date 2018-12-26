x
### Current status
* sourced most of the desired data
* cleaning data and settling on data model
  * found common keys to combine data from different sources
  * extracted some desired fields, e.g. day of week
  * settled on 6 tables with fields
  * casted numpy types to python native types and stored as Spark DataFrame
  * write as table from Spark
  * next steps
    * sort out home/away for lines table 

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
   * Travel: euclidian distances between cities? timezone change (-3 to +3)
  * use other sites' picks?
  * how many days ago was last game? covers by week too
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

#### open questions
  * how to handle early in the season vs. later? maybe start out building model for weeks 8-17?
  * how to handle playoffs vs. regular season?
  * how to figure out how to incorporate injuries
