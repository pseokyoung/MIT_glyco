##
# Contributors: Théo MAURI et al. (2021)
# 
# Contact: theo.mauri _[At]_ univ-lille.fr or guillaume.brysbaert _[At]_ univ-lille.fr
# 
# This script is a R script whose purpose is to run machine learning algorithms
# on a new dataset (available on the same git repository), creating training and testing datasets,
# models and tables with the different performance measurements.
# 
# Tested with R 3.6.3
#
##

GBT=FALSE ### Change here if you want to train gbt models (increase the running time)
	  nbCores=3 ### You can change it depending on the number of cores you want to use. It will also change the number of models produced.
          
          library(parallel)
          #install.packages("dplyr")
          cl<-makeCluster(nbCores)
          clusterEvalQ(cl,library(dplyr))
          #install.packages("readr")
          clusterEvalQ(cl,library(readr))
          #install.packages("randomForest")
          clusterEvalQ(cl,library(randomForest))
          #install.packages("smotefamily")
          clusterEvalQ(cl,library(smotefamily))
            #install.packages("caret")
          clusterEvalQ(cl,library(caret))
            #install.packages("ROSE")
          clusterEvalQ(cl,library(ROSE))
          #install.packages("e1071")
          clusterEvalQ(cl,library(e1071))
          #install.packages("xgboost")
          clusterEvalQ(cl,library(xgboost))
          
          #library(SparkR)
          #setwd("/home/your/path/to/OGP") ### Change the name of your path in case of Rstudio
          
          data_pos=read.csv(file = "file_for_ML_pos.csv",header = TRUE, sep=",")
          data_neg=read.csv(file = "file_for_ML_neg.csv",header = TRUE, sep=",")
          
          len_pos=(dim(data_pos)[1])
          len_neg=(dim(data_neg)[1])
          
          data_pos_total=data_pos[c(1:len_pos),c(1:15)]
          data_neg_total=data_neg[c(1:len_neg),c(1:15)]

          ### Data are put in on the different cluster###
          clusterExport(cl, "data_pos_total")
          clusterExport(cl, "data_neg_total")
          
          ### Vectors that will countain our performance results are set up. First part for non undersampled data test and the second part for the undersampled one ###
          i=0
          specific=0
          PPV=0
          vect_sensi_RF=c()
          vect_PPV_RF=c()
          vect_sensi_svm=c()
          vect_PPV_svm=c()
          vect_sensi_poly=c()
          vect_PPV_poly=c()
          vect_sensi_svm_radial=c()
          vect_PPV_svm_radial=c()
          vect_sensi_svm_sigmoid=c()
          vect_PPV_svm_sigmoid=c()
          vect_sensi_svm_sigmoid_tuned=c()
          vect_PPV_svm_sigmoid_tuned=c()
          vect_sensi_gbt=c()
          vect_PPV_gbt=c()
          vect_run=c()
          
          
          vect_sensi_RF_under=c()
          vect_PPV_RF_under=c()
          vect_sensi_svm_under=c()
          vect_PPV_svm_under=c()
          vect_sensi_poly_under=c()
          vect_PPV_poly_under=c()
          vect_sensi_svm_radial_under=c()
          vect_PPV_svm_radial_under=c()
          vect_sensi_svm_sigmoid_under=c()
          vect_PPV_svm_sigmoid_under=c()
          vect_sensi_svm_sigmoid_tuned_under=c()
          vect_PPV_svm_sigmoid_tuned_under=c()
          vect_sensi_gbt_under=c()
          vect_PPV_gbt_under=c()
          
          
          vect_RF_names<-c()
          vect_GBT_names<-c()
          vect_SVM_lin_names<-c()
          vect_SVM_Rad_names<-c()
          vect_SVM_Poly_names<-c()
          vect_SVM_Sig_names<-c()
          vect_SVM_Sig_Tuned_names<-c()
          #### Data train are created on each cluster ###
            
            
            ## Positive and negative data in 80/20 and retrieve the 80 and bind them together in data train ##
          clusterEvalQ(cl, data_train_pos<-data_pos_total %>% sample_frac(0.8)) ### 80% dpostive data
          clusterEvalQ(cl, data_train_neg<-data_neg_total %>% sample_frac(0.8)) ### 80% negativ data
          clusterEvalQ(cl,data_train<-rbind(data_train_pos,data_train_neg))
          clusterEvalQ(cl,levels(data_train$nb_ST))  
            ## Transforming the features of data_train into factors ##
          
          clusterEvalQ(cl,levels(data_train$nb_ST))
          clusterEvalQ(cl,levels(data_train$class)<-c(1:2))
          clusterEvalQ(cl,levels(data_train$nb_ST)<-c(0:21))
          clusterEvalQ(cl,levels(data_train$cpt_ali)<-c(0:3))
          clusterEvalQ(cl,levels(data_train$cpt_pos)<-c(0:3))
          clusterEvalQ(cl,levels(data_train$Pro_1)<-c(0:1))
          clusterEvalQ(cl,levels(data_train$min1)<-c(0:7))
          clusterEvalQ(cl,levels(data_train$plus1)<-c(0:7))
          clusterEvalQ(cl,levels(data_train$plus2)<-c(0:7))
          clusterEvalQ(cl,levels(data_train$plus3)<-c(0:7))
          clusterEvalQ(cl,levels(data_train$plus4)<-c(0:7))
          clusterEvalQ(cl,levels(data_train$plus5)<-c(0:7))
          clusterEvalQ(cl,levels(data_train$flexibility)<-seq(0,1, by = 0.001))
          clusterEvalQ(cl,levels(data_train$naturesite)<-c(0:1))
          clusterEvalQ(cl,levels(data_train$ss)<-c(0:2))
          clusterEvalQ(cl,levels(data_train$phi_psi)<-c(0:2))
          
          clusterEvalQ(cl,data_train<-transform(data_train,
                                                class=as.factor(class),
                                                nb_ST=as.factor(nb_ST),
                                                cpt_ali = as.factor(cpt_ali),
                                                cpt_pos=as.factor(cpt_pos),
                                                Pro_1=as.factor(Pro_1),
                                                min1=as.factor(min1),
                                                plus1=as.factor(plus1),
                                                plus2=as.factor(plus2),
                                                plus3=as.factor(plus3),
                                                plus4=as.factor(plus4),
                                                plus5=as.factor(plus5),
                                                naturesite=as.factor(naturesite),
                                                ss=as.factor(ss),
                                                phi_psi=as.factor(phi_psi)))
          
           ## Positive data are oversampled with ROSE algorithm ##
          
          #########################################################################################
          clusterEvalQ(cl,data_rose<-ROSE(class~., data_train,N=64434))
          clusterEvalQ(cl,(data_rose<-data_rose$data))
          clusterEvalQ(cl,levels(data_rose$class)<-c(1:2))
          clusterEvalQ(cl,levels(data_rose$nb_ST)<-c(0:21))
          clusterEvalQ(cl,levels(data_rose$cpt_ali)<-c(0:3))
          clusterEvalQ(cl,levels(data_rose$cpt_pos)<-c(0:3))
          clusterEvalQ(cl,levels(data_rose$Pro_1)<-c(0:1))
          clusterEvalQ(cl,levels(data_rose$min1)<-c(0:7))
          clusterEvalQ(cl,levels(data_rose$plus1)<-c(0:7))
          clusterEvalQ(cl,levels(data_rose$plus2)<-c(0:7))
          clusterEvalQ(cl,levels(data_rose$plus3)<-c(0:7))
          clusterEvalQ(cl,levels(data_rose$plus4)<-c(0:7))
          clusterEvalQ(cl,levels(data_rose$plus5)<-c(0:7))
          clusterEvalQ(cl,levels(data_rose$flexibility)<-seq(0,1, by = 0.001))
          clusterEvalQ(cl,levels(data_rose$naturesite)<-c(0:1))
          clusterEvalQ(cl,levels(data_rose$ss)<-c(0:2))
          clusterEvalQ(cl,levels(data_rose$phi_psi)<-c(0:2))
          
          data_rose<-clusterEvalQ(cl,(data_rose))
          
          
          
          ##########################################################################################
          ### Once our training data are ready we can build a model for each method on these training data ###
          ## First the Random Forest ##
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_RF_models<-clusterEvalQ(cl,name_model<-paste0("model_RF",x,".RData"))
          randomForestModels<-clusterEvalQ(cl,model_RF <- randomForest(class ~ ., data = data_rose, ntree = 200, na.action = na.omit, mtry=39,splitrule = extratrees,min.node.size = 1))
          clusterEvalQ(cl,save(model_RF,file = name_model ))

	  print("Models RF: Done")
          
          ## Second the Gradient Boosting Tree ##
	  if (GBT==TRUE){
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_GBT_models<-clusterEvalQ(cl,name_model<-paste0("model_GBT",x,".RData"))
          gradientBoostingTreeModels<-clusterEvalQ(cl,model_gbt <- train(class ~ ., data = data_rose, method= "xgbTree"))
          clusterEvalQ(cl,save(model_gbt,file = name_model ))
          
	        print("Models GBT: Done")
	  }

          ## Third the SVM ##
           #Linear
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_SVM_Linear_models<-clusterEvalQ(cl,name_model<-paste0("model_SVM_Linear",x,".RData"))
          linearSVMModels<-clusterEvalQ(cl,model_svm <- e1071::svm(class ~ ., data=data_rose,na.action =
                                    na.omit, scale = TRUE , kernel = "linear" ))
          clusterEvalQ(cl,save(model_svm,file = name_model ))

	  print("Models SVM-Linear: Done")
          
           # Polynomial
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_SVM_Poly_models<-clusterEvalQ(cl,name_model<-paste0("model_SVM_Poly",x,".RData"))
          polySVMModels<-clusterEvalQ(cl,model_svm_polynomial <- e1071::svm(class ~ ., data=data_rose,na.action =
                                               na.omit, scale = TRUE, kernel = "polynomial" ))
          clusterEvalQ(cl,save(model_svm_polynomial,file = name_model ))
          
	  print("Models SVM-Polynomial: Done")

           # Radial basis
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_SVM_Radial_models<-clusterEvalQ(cl,name_model<-paste0("model_SVM_Radial",x,".RData"))
          radialBasisSVMModels<-clusterEvalQ(cl,model_svm_radial <- e1071::svm(class ~ ., data=data_rose,na.action =
                                           na.omit, scale = TRUE , kernel = "radial" ))
          clusterEvalQ(cl,save(model_svm_radial,file = name_model ))
          
	  print("SVM-Radial: Done")

           # Sigmoid without hyper parameters
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_SVM_Sig_models<-clusterEvalQ(cl,name_model<-paste0("model_SVM_Sigmoid",x,".RData"))
          sigmoidSVMModels<-clusterEvalQ(cl,model_svm_sigmoid <- e1071::svm(class ~., data=data_rose,na.action =
                                            na.omit, scale = TRUE, kernel = "sigmoid"))
          clusterEvalQ(cl,save(model_svm_sigmoid,file = name_model ))
          
	  print("Models SVM-Sigmoid: Done")

          # Sigmoid with hyper parameters
          clusterEvalQ(cl,x<-sample(1:1000, 1))
          all_SVM_Sig_Tuned_models<-clusterEvalQ(cl,name_model<-paste0("model_SVM_Sigmoid_Tuned",x,".RData"))
          sigmoidWithHPSVMModels<-clusterEvalQ(cl,model_svm_sigmoid_tuned <- e1071::svm(class ~., data=data_rose,na.action =
                                                  na.omit, scale = TRUE, kernel = "sigmoid", gamma= 1, cost = 4))
          clusterEvalQ(cl,save(model_svm_sigmoid_tuned,file = name_model ))
          
          ### Once models are created we can create the dataset with the remaining 20% ###
           ## Without undersampling
          clusterEvalQ(cl,data_test_pos<- anti_join(data_pos_total, data_train_pos))
          clusterEvalQ(cl,data_test_neg<- anti_join(data_neg_total, data_train_neg))
          clusterEvalQ(cl,data_test<-rbind(data_test_neg,data_test_pos))


         
        
          clusterEvalQ(cl, data_test_pos<-anti_join(data_pos_total, data_train_pos)) ### 80% postive data
          clusterEvalQ(cl, data_test_neg<- anti_join(data_neg_total, data_train_neg)) ### 80% negativ data
          clusterEvalQ(cl,data_test<-rbind(data_test_pos,data_test_neg))
          clusterEvalQ(cl,dim(data_test))
          
            # Features transformed into factors as for training set
          clusterEvalQ(cl,data_test<-transform(data_test, 
                                 class=as.factor(class),
                                 nb_ST=as.factor(nb_ST),
                                 cpt_ali = as.factor(cpt_ali),
                                 cpt_pos=as.factor(cpt_pos),
                                 Pro_1=as.factor(Pro_1),
                                 min1=as.factor(min1),
                                 plus1=as.factor(plus1),
                                 plus2=as.factor(plus2),
                                 plus3=as.factor(plus3),
                                 plus4=as.factor(plus4),
                                 plus5=as.factor(plus5),
                                 naturesite=as.factor(naturesite),
                                 ss=as.factor(ss),
                                 phi_psi=as.factor(phi_psi)))
          clusterEvalQ(cl,levels(data_test$nb_ST))
          clusterEvalQ(cl,data_test$class<-as.factor(data_test$class))
          clusterEvalQ(cl,levels(data_test$nb_ST))
          
          print("data test is reading")
           ## with undersampling
          clusterEvalQ(cl,data_test_under_tmp<-ovun.sample(class ~., data_test, method=c("under")))
          clusterEvalQ(cl,data_test_under<-data_test_under_tmp$data)
          
          clusterEvalQ(cl,levels(data_test$class)<-c(1:2))
          clusterEvalQ(cl,levels(data_test$nb_ST)<-c(0:21))
          clusterEvalQ(cl,levels(data_test$cpt_ali)<-c(0:3))
          clusterEvalQ(cl,levels(data_test$cpt_pos)<-c(0:3))
          clusterEvalQ(cl,levels(data_test$Pro_1)<-c(0:1))
          clusterEvalQ(cl,levels(data_test$min1)<-c(0:7))
          clusterEvalQ(cl,levels(data_test$plus1)<-c(0:7))
          clusterEvalQ(cl,levels(data_test$plus2)<-c(0:7))
          clusterEvalQ(cl,levels(data_test$plus3)<-c(0:7))
          clusterEvalQ(cl,levels(data_test$plus4)<-c(0:7))
          clusterEvalQ(cl,levels(data_test$plus5)<-c(0:7))
          clusterEvalQ(cl,levels(data_test$flexibility)<-seq(0,1, by = 0.001))
          clusterEvalQ(cl,levels(data_test$naturesite)<-c(0:1))
          clusterEvalQ(cl,levels(data_test$ss)<-c(0:2))
          clusterEvalQ(cl,levels(data_test$phi_psi)<-c(0:2))
          
          clusterEvalQ(cl,levels(data_test_under$class)<-c(1:2))
          clusterEvalQ(cl,levels(data_test_under$nb_ST)<-c(0:21))
          clusterEvalQ(cl,levels(data_test_under$cpt_ali)<-c(0:3))
          clusterEvalQ(cl,levels(data_test_under$cpt_pos)<-c(0:3))
          clusterEvalQ(cl,levels(data_test_under$Pro_1)<-c(0:1))
          clusterEvalQ(cl,levels(data_test_under$min1)<-c(0:7))
          clusterEvalQ(cl,levels(data_test_under$plus1)<-c(0:7))
          clusterEvalQ(cl,levels(data_test_under$plus2)<-c(0:7))
          clusterEvalQ(cl,levels(data_test_under$plus3)<-c(0:7))
          clusterEvalQ(cl,levels(data_test_under$plus4)<-c(0:7))
          clusterEvalQ(cl,levels(data_test_under$plus5)<-c(0:7))
          clusterEvalQ(cl,levels(data_test_under$flexibility)<-seq(0,1, by = 0.001))
          clusterEvalQ(cl,levels(data_test_under$naturesite)<-c(0:1))
          clusterEvalQ(cl,levels(data_test_under$ss)<-c(0:2))
          clusterEvalQ(cl,levels(data_test_under$phi_psi)<-c(0:2))
          
          print("data test undersampled is reading")

          
          ### Then we predict with each model on both test data set (under and not undersampled) ###
          
          ## RF 
          #not undersampled
          
          clusterEvalQ(cl,data_test$predicted_RF<-predict(model_RF,data_test))
          conf_all_features_RF<-clusterEvalQ(cl,conf_all_features_RF <- confusionMatrix(data = factor(data_test$predicted_RF), reference = factor(data_test$class), positive = "1"))
          data_test= clusterEvalQ(cl,data_test)
          data_train= clusterEvalQ(cl,data_train)
          
          print("Prediction real proportion RF: Done")
          # undersampled
          clusterEvalQ(cl,data_test_under$predicted_RF<-predict(model_RF,data_test_under))
          print("Prediction undersampled RF: Done")
          conf_all_features_RF_under<-clusterEvalQ(cl,conf_all_features_RF_under <- confusionMatrix(data = factor(data_test_under$predicted_RF), reference = factor(data_test_under$class), positive = "1"))
          print("Confusion matrices RF: Done")
          
          if (GBT == TRUE){
            ## GBT
            #not undersampled
            clusterEvalQ(cl,data_test$gbt_predicted<-predict(model_gbt,data_test))
            conf_all_features_gbt<-clusterEvalQ(cl,conf_all_features_gbt <- confusionMatrix(data = factor(data_test$gbt_predicted), reference = factor(data_test$class), positive = "1"))
            print("Prediction real proportion GBT: Done")
            #undersampled
            clusterEvalQ(cl,data_test_under$gbt_predicted<-predict(model_gbt,data_test_under))
            print("Prediction undersampled GBT: Done")
            conf_all_features_gbt_under<-clusterEvalQ(cl,conf_all_features_gbt_under <- confusionMatrix(data = factor(data_test_under$gbt_predicted), reference = factor(data_test_under$class), positive = "1"))
            print("Confusion matrices GBT: Done")
            }
          
          ## SVM Linear
          #not undersampled
          clusterEvalQ(cl,data_test$svm_predicted<-predict(model_svm,data_test))
          conf_all_features_svm<-clusterEvalQ(cl,conf_all_features_svm <- confusionMatrix(data = factor(data_test$svm_predicted), reference = factor(data_test$class), positive = "1"))
          print("Prediction real proportion SVM Linear: Done")
          #undersampled
          clusterEvalQ(cl,data_test_under$svm_predicted<-predict(model_svm,data_test_under))
          conf_all_features_svm_under<-clusterEvalQ(cl,conf_all_features_svm_under <- confusionMatrix(data = factor(data_test_under$svm_predicted), reference = factor(data_test_under$class), positive = "1"))
          print("Prediction undersampled SVM Linear: Done")
          print("Confusion matrices SVM Linear: Done")
          ## SVM Poly
          #not undersampled
          clusterEvalQ(cl,data_test$svm_predicted_poly<-predict(model_svm_polynomial,data_test))
          print("Prediction real proportion SVM Polynomial: Done")
          conf_all_features_svm_poly<-clusterEvalQ(cl,conf_all_features_svm_poly <- confusionMatrix(data = factor(data_test$svm_predicted_poly), reference = factor(data_test$class), positive = "1"))
          
          #undersampled
          clusterEvalQ(cl,data_test_under$svm_predicted_poly<-predict(model_svm_polynomial,data_test_under))
          conf_all_features_svm_poly_under<-clusterEvalQ(cl,conf_all_features_svm_poly_under <- confusionMatrix(data = factor(data_test_under$svm_predicted_poly), reference = factor(data_test_under$class), positive = "1"))
          print("Prediction undersampled SVM Polynomial: Done")
          print("Confusion matrices SVM Polynomial: Done")
          ## SVM Radial basis
          #not undersampled
          clusterEvalQ(cl,data_test$svm_predicted_rad<-predict(model_svm_radial,data_test))
          print("Prediction real proportion SVM Radial: Done")
          conf_all_features_svm_radial<-clusterEvalQ(cl,conf_all_features_svm_radial <- confusionMatrix(data = factor(data_test$svm_predicted_rad), reference = factor(data_test$class), positive = "1"))
          
          #undersampled
          clusterEvalQ(cl,data_test_under$svm_predicted_rad<-predict(model_svm_radial,data_test_under))
          conf_all_features_svm_radial_under<-clusterEvalQ(cl,conf_all_features_svm_radial_under <- confusionMatrix(data = factor(data_test_under$svm_predicted_rad), reference = factor(data_test_under$class), positive = "1"))
          print("Prediction undersampled SVM Radial: Done")
          print("Confusion matrices SVM Radial: Done")
          ## SVM Sigmoid
          #not undersampled
          clusterEvalQ(cl, data_test$svm_predicted_sig<-predict(model_svm_sigmoid,data_test))
          print("Prediction real proportion SVM Sigmoid: Done")
          conf_all_features_svm_sig<-clusterEvalQ(cl, conf_all_features_svm_sig <- confusionMatrix(data = factor(data_test$svm_predicted_sig), reference = factor(data_test$class), positive = "1"))
          
          #undersampled
          clusterEvalQ(cl,data_test_under$svm_predicted_sig<-predict(model_svm_sigmoid,data_test_under))
          conf_all_features_svm_sig_under<-clusterEvalQ(cl,conf_all_features_svm_sig_under <- confusionMatrix(data = factor(data_test_under$svm_predicted_sig), reference = factor(data_test_under$class), positive = "1"))
          print("Prediction undersampled SVM Sigmoid: Done")
          print("Confusion matrices SVM Sigmoid: Done")
          ## SVM Sigmoid with Hyper Parameters
          #not undersampled
          clusterEvalQ(cl, data_test$svm_predicted_sig_tuned<-predict(model_svm_sigmoid_tuned,data_test))
          conf_all_features_svm_sig_tuned<-clusterEvalQ(cl, conf_all_features_svm_sig_tuned <- confusionMatrix(data = factor(data_test$svm_predicted_sig_tuned), reference = factor(data_test$class), positive = "1"))
          print("Prediction undersampled SVM Sigmoid tuned: Done")
          #undersampled
          clusterEvalQ(cl,data_test_under$svm_predicted_sig_tuned<-predict(model_svm_sigmoid_tuned,data_test_under))
          conf_all_features_svm_sig_tuned_under<-clusterEvalQ(cl,conf_all_features_svm_sig_tuned_under <- confusionMatrix(data = factor(data_test_under$svm_predicted_sig_tuned), reference = factor(data_test_under$class), positive = "1"))
          print("Prediction undersampled SVM Sigmoid tuned: Done")
          print("Confusion matrices SVM Sigmoid tuned: Done")
           
          
          ### Now all the statistics (PPV and sensitivity) are retrieved from our results and put in a data frame ###
          i=1
          
          while (i < (nbCores+1)){  #to browse all the results in the different cores
            vect_run=c(vect_run,i)
            
            vect_RF_names<-c(vect_RF_names,all_RF_models[[i]])
            vect_sensi_RF=c(vect_sensi_RF,conf_all_features_RF[[i]]$byClass[1])
            vect_PPV_RF=c(vect_PPV_RF,conf_all_features_RF[[i]]$byClass[3])
            if (GBT == TRUE){
            vect_sensi_gbt=c(vect_sensi_gbt,conf_all_features_gbt[[i]]$byClass[1])
            vect_PPV_gbt=c(vect_PPV_gbt,conf_all_features_gbt[[i]]$byClass[3])
            vect_GBT_names<-c(vect_GBT_names,all_GBT_models[[i]])
            }
            
            vect_sensi_svm=c(vect_sensi_svm,conf_all_features_svm[[i]]$byClass[1])
            vect_PPV_svm=c(vect_PPV_svm,conf_all_features_svm[[i]]$byClass[3])
            vect_SVM_lin_names<-c(vect_SVM_lin_names,all_SVM_Linear_models[[i]])
            
            vect_sensi_poly=c(vect_sensi_poly,conf_all_features_svm_poly[[i]]$byClass[1])
            vect_PPV_poly=c(vect_PPV_poly,conf_all_features_svm_poly[[i]]$byClass[3])
            vect_SVM_Poly_names<-c(vect_SVM_Poly_names,all_SVM_Poly_models[[i]])
            
            vect_sensi_svm_radial=c(vect_sensi_svm_radial,conf_all_features_svm_radial[[i]]$byClass[1])
            vect_PPV_svm_radial=c(vect_PPV_svm_radial,conf_all_features_svm_radial[[i]]$byClass[3])
            vect_SVM_Rad_names<-c(vect_SVM_Rad_names,all_SVM_Radial_models[[i]])
            
            vect_sensi_svm_sigmoid=c(vect_sensi_svm_sigmoid,conf_all_features_svm_sig[[i]]$byClass[1])
            vect_PPV_svm_sigmoid=c(vect_PPV_svm_sigmoid,conf_all_features_svm_sig[[i]]$byClass[3])
            vect_SVM_Sig_names<-c(vect_SVM_Sig_names,all_SVM_Sig_models[[i]])
            
            vect_sensi_svm_sigmoid_tuned=c(vect_sensi_svm_sigmoid_tuned,conf_all_features_svm_sig_tuned[[i]]$byClass[1])
            vect_PPV_svm_sigmoid_tuned=c(vect_PPV_svm_sigmoid_tuned,conf_all_features_svm_sig_tuned[[i]]$byClass[3])
            vect_SVM_Sig_Tuned_names<-c(vect_SVM_Sig_Tuned_names,all_SVM_Sig_Tuned_models[[i]])
            
            print("Performances real proportion: Done")
            ### Below: same for undersampled data
            
            vect_sensi_RF_under=c(vect_sensi_RF_under,conf_all_features_RF_under[[i]]$byClass[1])
            vect_PPV_RF_under=c(vect_PPV_RF_under,conf_all_features_RF_under[[i]]$byClass[3])
            if (GBT == TRUE){
            vect_sensi_gbt_under=c(vect_sensi_gbt_under,conf_all_features_gbt_under[[i]]$byClass[1])
            vect_PPV_gbt_under=c(vect_PPV_gbt_under,conf_all_features_gbt_under[[i]]$byClass[3])
            }
            vect_sensi_svm_under=c(vect_sensi_svm_under,conf_all_features_svm_under[[i]]$byClass[1])
            vect_PPV_svm_under=c(vect_PPV_svm_under,conf_all_features_svm_under[[i]]$byClass[3])
            vect_sensi_poly_under=c(vect_sensi_poly_under,conf_all_features_svm_poly_under[[i]]$byClass[1])
            vect_PPV_poly_under=c(vect_PPV_poly_under,conf_all_features_svm_poly_under[[i]]$byClass[3])
            vect_sensi_svm_radial_under=c(vect_sensi_svm_radial_under,conf_all_features_svm_radial_under[[i]]$byClass[1])
            vect_PPV_svm_radial_under=c(vect_PPV_svm_radial_under,conf_all_features_svm_radial_under[[i]]$byClass[3])
            vect_sensi_svm_sigmoid_under=c(vect_sensi_svm_sigmoid_under,conf_all_features_svm_sig_under[[i]]$byClass[1])
            vect_PPV_svm_sigmoid_under=c(vect_PPV_svm_sigmoid_under,conf_all_features_svm_sig_under[[i]]$byClass[3])
            vect_sensi_svm_sigmoid_tuned_under=c(vect_sensi_svm_sigmoid_tuned_under,conf_all_features_svm_sig_tuned_under[[i]]$byClass[1])
            vect_PPV_svm_sigmoid_tuned_under=c(vect_PPV_svm_sigmoid_tuned_under,conf_all_features_svm_sig_tuned_under[[i]]$byClass[3])
          
            i=i+1
            print(i)
          }
          print("Performances undersampled: Done")
          if (GBT == TRUE){
          DF_x_all_methods_no_under=data.frame(list(vect_run,
                                                     vect_RF_names,
                                                     vect_sensi_RF,
                                                     vect_PPV_RF,
                                                     vect_GBT_names,
                                                     vect_sensi_gbt,
                                                     vect_PPV_gbt,
                                                     vect_SVM_lin_names,
                                                     vect_sensi_svm,
                                                     vect_PPV_svm,
                                                     vect_SVM_Poly_names,
                                                     vect_sensi_poly,
                                                     vect_PPV_poly,
                                                     vect_SVM_Rad_names,
                                                     vect_sensi_svm_radial,
                                                     vect_PPV_svm_radial,
                                                     vect_SVM_Sig_names,
                                                     vect_sensi_svm_sigmoid,
                                                     vect_PPV_svm_sigmoid,
                                                     vect_SVM_Sig_Tuned_names,
                                                     vect_sensi_svm_sigmoid_tuned,
                                                     vect_PPV_svm_sigmoid_tuned)
          )
          colnames(DF_x_all_methods_no_under)<-c("run","name_model_RF","Sensi RF","PPV RF","name_model_GBT","Sensi GBT","PPV GBT","name_model_Lin","Sensi SVM Lin","PPV Lin","name_model_Poly","Sensi Poly","PPV Poly","name_model_Rad","Sensi Rad","PPV Rad","name_model_Sig","Sensi Sig","PPV Sig","name_model_Sig_Tuned","Sensi Sig Tuned","PPV Sig Tuned")
                                                  #
          print(DF_x_all_methods_no_under)
          
          DF_x_all_methods_test_under=data.frame(vect_run,
                                                       vect_RF_names,
                                                       vect_sensi_RF_under,
                                                       vect_PPV_RF_under,
                                                       vect_GBT_names,
                                                       vect_sensi_gbt_under,
                                                       vect_PPV_gbt_under,
                                                       vect_SVM_lin_names,
                                                       vect_sensi_svm_under,
                                                       vect_PPV_svm_under,
                                                       vect_SVM_Poly_names,
                                                       vect_sensi_poly_under,
                                                       vect_PPV_poly_under,
                                                       vect_SVM_Rad_names,
                                                       vect_sensi_svm_radial_under,
                                                       vect_PPV_svm_radial_under,
                                                       vect_SVM_Sig_names,
                                                       vect_sensi_svm_sigmoid_under,
                                                       vect_PPV_svm_sigmoid_under,
                                                       vect_SVM_Sig_Tuned_names,
                                                       vect_sensi_svm_sigmoid_tuned_under,
                                                       vect_PPV_svm_sigmoid_tuned_under
          )
          colnames(DF_x_all_methods_test_under)<-c("run","name_model_RF","Sensi RF","PPV RF","name_model_GBT","Sensi GBT","PPV GBT","name_model_Lin","Sensi SVM Lin","PPV Lin","name_model_Poly","Sensi Poly","PPV Poly","name_model_Rad","Sensi Rad","PPV Rad","name_model_Sig","Sensi Sig","PPV Sig","name_model_Sig_Tuned","Sensi Sig Tuned","PPV Sig Tuned")
          print(DF_x_all_methods_test_under)
          }
          if (GBT == FALSE){
            DF_x_all_methods_no_under=data.frame(list(vect_run,
                                                       vect_RF_names,
                                                       vect_sensi_RF,
                                                       vect_PPV_RF,
                                                       vect_SVM_lin_names,
                                                       vect_sensi_svm,
                                                       vect_PPV_svm,
                                                       vect_SVM_Poly_names,
                                                       vect_sensi_poly,
                                                       vect_PPV_poly,
                                                       vect_SVM_Rad_names,
                                                       vect_sensi_svm_radial,
                                                       vect_PPV_svm_radial,
                                                       vect_SVM_Sig_names,
                                                       vect_sensi_svm_sigmoid,
                                                       vect_PPV_svm_sigmoid,
                                                       vect_SVM_Sig_Tuned_names,
                                                       vect_sensi_svm_sigmoid_tuned,
                                                       vect_PPV_svm_sigmoid_tuned)
            )
            colnames(DF_x_all_methods_no_under)<-c("run","name_model_RF","Sensi RF","PPV RF","name_model_Lin","Sensi SVM Lin","PPV Lin","name_model_Poly","Sensi Poly","PPV Poly","name_model_Rad","Sensi Rad","PPV Rad","name_model_Sig","Sensi Sig","PPV Sig","name_model_Sig_Tuned","Sensi Sig Tuned","PPV Sig Tuned")
            #
            print(DF_x_all_methods_no_under)
            
            DF_x_all_methods_test_under=data.frame(vect_run,
                                                    vect_RF_names,
                                                    vect_sensi_RF_under,
                                                    vect_PPV_RF_under,
                                                    vect_SVM_lin_names,
                                                    vect_sensi_svm_under,
                                                    vect_PPV_svm_under,
                                                    vect_SVM_Poly_names,
                                                    vect_sensi_poly_under,
                                                    vect_PPV_poly_under,
                                                    vect_SVM_Rad_names,
                                                    vect_sensi_svm_radial_under,
                                                    vect_PPV_svm_radial_under,
                                                    vect_SVM_Sig_names,
                                                    vect_sensi_svm_sigmoid_under,
                                                    vect_PPV_svm_sigmoid_under,
                                                    vect_SVM_Sig_Tuned_names,
                                                    vect_sensi_svm_sigmoid_tuned_under,
                                                    vect_PPV_svm_sigmoid_tuned_under
            )
            colnames(DF_x_all_methods_test_under)<-c("run","name_model_RF","Sensi RF","PPV RF","name_model_Lin","Sensi SVM Lin","PPV Lin","name_model_Poly","Sensi Poly","PPV Poly","name_model_Rad","Sensi Rad","PPV Rad","name_model_Sig","Sensi Sig","PPV Sig","name_model_Sig_Tuned","Sensi Sig Tuned","PPV Sig Tuned")
            print(DF_x_all_methods_test_under)
          }
          name_no_under<-paste0("DF_",nbCores,"_runs_all_methods_no_under.RData")
          save(DF_x_all_methods_no_under, file = name_no_under)
          name_under<-paste0("DF_",nbCores,"_runs_all_methods_under.RData")
          save(DF_x_all_methods_test_under, file = name_under)
          stopCluster(cl)
