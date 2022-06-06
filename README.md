# AccuracyTest_And_FeatureImportance
Physical and non-physical properties (38 features, many of them are irrelevant) of dense pockets of gas (called dense cores) extracted from 5 simulations of colliding molecular clouds from Sakre et al. 2022 (in prep.)

Purpose: 1. To predict model from test cores (50 percent data) 
         2. To find importance of features (properties)
         
Method: 1. kNNeighbors (No. of neighbors = 5(default)) is used 
        2. Random Forest classifier (with max_depth=2) is used to judge feature
        importance
        
Results: 1. Found very low accuracy of 30 %, implying properties have weak 
        relation with their model type
         2. RFClassifier gave high feature importance to "x-position" feature and low importance to "y-velocity"
         
Does the Result make sense?

Yes.

1.Low Accuracy: The low accuracy was expected since many of the physical conditions are common and thus it is           hard to classify. 

2.1. x-position high feature importance: Based on collision speed and cloud size, the x-position can be highly    
                                         simulation model dependent.
                                    
2.2. y-velocity high feature importance: We expect no specific preffered y-velocity value due to 
                                        near symmetry of all our 5 collision models along y = 0 plane
                                      
