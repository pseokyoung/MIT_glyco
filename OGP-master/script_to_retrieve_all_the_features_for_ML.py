##
# Contributors: Théo MAURI et al. (2021)
# 
# Contact: theo.mauri _[At]_ univ-lille.fr or guillaume.brysbaert _[At]_ univ-lille.fr
# 
# This script is a Python script whose purpose is to retrieve features for machine learning algorithms.
# These features are retrieved from a dataset (csv file) and results are gathered in a csv file.
# 
# Tested with Python 3.6
#
##  

import sys, os, xlrd, csv, re
from xlwt import Workbook
fileout=open("file_for_ML.csv", "w")

#### Here are setting functions ####
def checkAAPolarity(letter):
    if (letter == "S" or letter == "T"):
        return "A"
    elif (letter == "N" or letter == "Q"):
        return "B"
    elif (letter == "R" or letter == "K" or letter == "H"):
        return "C"
    elif (letter == "E" or letter == "D"):
        return "D"
    elif (letter == "M" or letter == "C"):
        return "E"
    elif(letter == "Y" or letter == "F" or letter == "W"):
        return "F"
    elif(letter == "A" or letter == "V" or letter == "L" or letter == "I" or letter == "P"):
        return "G"
    elif(letter == "_"):
        return "Z" # For empty
    else:
        return "H"

def checkAALengthClass(letter):
    if (letter == "R" or letter == "K"):
        return 5
    elif (letter == "V" or letter == "A"):
        return 2
    elif (letter == "D" or letter == "E" or letter == "N" or letter == "Q" or letter == "M"):
        return 4
    elif (letter == "S" or letter == "I" or letter == "L" or letter == "T" or letter == "C"):
        return 3
    elif (letter == "F" or letter == "W" or letter == "Y" or letter == "H"):
        return 6
    elif(letter == "G"):
        return 1
    elif(letter == "_"):
        return 0 # For empty
    else:
        return 7
    
def windowMaking(posbef,posaft,psite,seq): #Will return a list from the posbef to posaft the psite of the sequence
    if type(seq)==list:
        kk=1
    else:
        seq=list(seq)
    firstAA=psite-posbef
    lastAA=psite+posaft+1
    window=[]
    i=firstAA
    if i<0:
        while i <0:
            window.append("_")
            i=i+1
        i=0
    while i < lastAA:
        try:
            seq[i]
            seq[i]!="\n"
            window.append(seq[i])
        except:
            
            window.append("_")
        i=i+1
    return window 


### Here starts the script ###

with open('all_sites.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    prot_name="a"
    for row in reader:
        if row[0]==prot_name:
            continue
        prot_name=row[0]
        print(prot_name)
        
       
        prot_name_file_flexibility="./dynamine_results/"+prot_name+"_backbone.pred"
        #print(prot_name_file_flexibility)
        if (os.path.isfile(prot_name_file_flexibility)) == False:
            continue
        flexibility_file=open(prot_name_file_flexibility,"r")
        data_flex=flexibility_file.readlines()
        
        index=int(row[1])-1 ### -1 because the file reading starts at 0 ###
        ##################################################################################
        ### This part is to retrieve the flexibility score for the O-GlcNAcylated site ###
        ##################################################################################
        index_flexibility=index+11
        if index_flexibility > len(data_flex):
            continue
        else:
            line_flex=data_flex[index_flexibility].split()
            score_flex=line_flex[1]
        ###################################################################################
        tabseq=list(row[2])
        name=row[0]
        i=0
        while i < len(tabseq):
            #####################################################################
            ### This part is to retrieve the features for O-GlcNAcylated sites ##
            #####################################################################
            if tabseq[i]=="S" or tabseq[i]=="T":
                nom_fichier="./spider3_results/"+prot_name+".spd33"
                if i == index:
                    with open('all_sites_with_colnames.csv', mode='r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        line_count = 0
                        site=[]
                        for row in csv_reader:
                            if(str(row["name"]))==prot_name:
                                site.append(int(row["site"])-1)
   
                    fichier=open(nom_fichier,"r")
                    data=fichier.read().splitlines()
    
                    for v in site:
                        print(v)
        
                        list_value_phi=[]
                        list_value_psi=[]
                        siite=int(v)+1
                        rang=range((siite-3),(siite+2))
                        for m in rang:
           
                            if m <=0:
                                continue
                            else:
                
                                line=(data[m]).split()
                                list_value_phi.append(float(line[4]))
                                list_value_psi.append(float(line[5]))
                        cpt_beta=0
                        cpthel=0
                        other=0
                        j=0
                        value_psh_psi=0
                        while j < len(list_value_phi):
                            #print(j)
                            if list_value_phi[j] <(-50) and list_value_phi[j] > (-160):
                                if list_value_psi[j] < (180) and list_value_psi[j] > (100):
                    
                                    cpt_beta=cpt_beta+1
            
                                else:
                                    if (list_value_psi[j] <= 20) and (list_value_psi[j]>=-60):
                                        cpthel=cpthel+1
                        
                                    else: 
                                        other=other+1
                            j=j+1
                        if cpt_beta > cpthel and cpt_beta > other:
                            value_psh_psi=1 ### 1 = beta
                        elif cpthel > cpt_beta and cpthel > other:
                            value_psh_psi=2 #### 2 = alpha
                       
                       
                            
                        line=(data[v]).split()
                        if line[2] == "C":
                            ss=0
                        elif line[2] == "H":
                            ss=1
                        elif line[2] == "E":
                            ss=2
                        Window=windowMaking(10,10,v,tabseq)
                        pos_or_neg = "1"
                        cpt=-10
                        cptST=0  # Will be used to count the number of ser and thr in the window 
                        cpt_ali=0 # Will be used to count the number of aliphatic residues in the window between -3 to -1 
                        cpt_pos=0 # Will be used to count the number of positively charged residues in the window between -7 to -5 
                        Pro_1= 0 # To know if there is a proline just after the O-GlcNAcylated site
                        for j in Window:
                        #print(j)
                            if j == "S" or j== "T":
                                cptST=cptST+1
                            if cpt== -3 or cpt==-2 or cpt==-1:
                                if checkAAPolarity(j) == 'G':
                                    cpt_ali=cpt_ali+1
                            if cpt== -7 or cpt==-6 or cpt==-5:
                                if checkAAPolarity(j) == 'C':
                                    cpt_pos=cpt_pos+1
                        ### This part is to see the composition in side chain size class from -1 to +5 ###
                            if cpt == -1:
                                min1 = checkAALengthClass(j)
                            if cpt == 1:
                                plus1 = checkAALengthClass(j)
                            if cpt == 2:
                                plus2 = checkAALengthClass(j)
                            if cpt == 3:
                                plus3 = checkAALengthClass(j)
                            if cpt == 4:
                                plus4 = checkAALengthClass(j)
                            if cpt == 5:
                                plus5 = checkAALengthClass(j)
                            if cpt==1 and j == "P":
                                Pro_1=1
                        ### Next is to retrieve the nature of the site (Ser or Thr) ###
                            if cpt==0:
                                if j=="S":
                                    site_nature=0
                                elif j=="T":
                                    site_nature=1
                                
                            cpt=cpt+1
                        
                        fileout.write(str(pos_or_neg)+","+str(cptST)+","+str(cpt_ali)+","+str(cpt_pos)+","+str(Pro_1)+","+str(min1)+","+str(plus1)+","+str(plus2)+","+str(plus3)+","+str(plus4)+","+str(plus5)+","+str(score_flex)+","+str(site_nature)+","+str(ss)+","+str(value_psh_psi)+"\n")
                        
                    ##########################################################################
                    ### This part is to retrieve the features for non O-GlcNAcylated sites ###
                    ##########################################################################
                else:
                    flexibility_file=open(prot_name_file_flexibility,"r")
                    data_flex=flexibility_file.readlines()
                    ######################################################################################
                    ### This part is to retrieve the flexibility score for the non O-GlcNAcylated site ###
                    ######################################################################################
                    index_flexibility=i+11
                    if index_flexibility > len(data_flex):
                        continue
                    elif index_flexibility == len(data_flex):
                        print("\n")
                    else:
                        line_flex=data_flex[index_flexibility].split()
                        score_flex=line_flex[1]
                    with open('all_sites_with_colnames.csv', mode='r') as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        line_count = 0
                        site=[]
                        for row in csv_reader:
                            if(str(row["name"]))==prot_name:
                                site.append(int(row["site"]))

                    fichier=open(nom_fichier,"r")
                    data=fichier.read().splitlines()

                    for v in site:

                        
                        b=1
                        while b <len(data):
                            if b==v:
                                b=b+1

                            else:
                                list_value_phi=[]
                                list_value_psi=[]
                                siite=int(b)
                                rang=range((siite-3),(siite+3)) 
                                for k in rang:
                                    if k <=0:
                                        k=1
                                    elif k>=len(data):
                                        continue
                                    else:

                                        line=(data[k]).split() #+1 because the first line is the file description
                                        list_value_phi.append(float(line[4]))
                                        list_value_psi.append(float(line[5]))
                                    cpt_beta=0
                                    cpthel=0
                                    other=0
                                    j=0
                                    value_psh_psi=0
                                    while j < len(list_value_phi):

                                        if list_value_phi[j] <(-50) and list_value_phi[j] > (-160):
                                            if list_value_psi[j] < (180) and list_value_psi[j] > (100):
                    
                                                 cpt_beta=cpt_beta+1
            
                                            else:
                                                if (list_value_psi[j] <= 20) and (list_value_psi[j]>=-60):
                                                    cpthel=cpthel+1
                        
                                                else: 
                                                    other=other+1
                                        
                                        j=j+1
                                    if cpt_beta > cpthel and cpt_beta > other:
                                        value_psh_psi=1 ### 1 = beta
                                    elif cpthel > cpt_beta and cpthel > other:
                                        value_psh_psi=2 #### 2 = alpha
                                line=(data[b]).split()
                                if line[1]== "S" or line[1] == "T":
                                    if line[2] == "C":
                                        ss=0
                                    elif line[2] == "H":
                                        ss=1
                                    elif line[2] == "E":
                                        ss=2
                   
                                    b=b+1
                                else:
                                    b=b+1
                    Window=windowMaking(10,10,i,tabseq,)
                    index_flex_neg=i+11
                    if index_flex_neg >= len(data_flex):
                        i=i+1
                        continue
                    line_flex_neg=data_flex[i+11].split()
                    score_flex_neg=line_flex_neg[1]
                    for j in Window:
                        pos_or_neg=2
                        cpt=-10
                        cptST=0  # Will be used to count the number of ser and thr in the window 
                        cpt_ali=0 # Will be used to count the number of aliphatic residues in the window between -3 to -1 
                        cpt_pos=0 # Will be used to count the number of positively charged residues in the window between -7 to -5 
                        Pro_1= 0  # To know if there is a proline just after the O-GlcNAcylated site
                      
                        for j in Window:
                        #print(j)
                            
                            
                            if j == "S" or j== "T":
                                cptST=cptST+1
                            if cpt== -3 or cpt==-2 or cpt==-1:
                                if checkAAPolarity(j) == 'G':
                                    cpt_ali=cpt_ali+1
                            if cpt== -7 or cpt==-6 or cpt==-5:
                                if checkAAPolarity(j) == 'C':
                                    cpt_pos=cpt_pos+1
                            ### This part is to see the composition in side chain size class form -1 to +5 ###
                            if cpt == -1:
                                min1 = checkAALengthClass(j)
                            if cpt == 1:
                                plus1 = checkAALengthClass(j)
                            if cpt == 2:
                                plus2 = checkAALengthClass(j)
                            if cpt == 3:
                                plus3 = checkAALengthClass(j)
                            if cpt == 4:
                                plus4 = checkAALengthClass(j)
                            if cpt == 5:
                                plus5 = checkAALengthClass(j)
                            if cpt==1 and j == "P":
                                Pro_1=1
                        ### Next is to retrieve the nature of the site (Ser or Thr) ###
                            if cpt==0:
                                if j=="S":
                                    site_nature=0
                                if j=="T":
                                    site_nature=1
                            cpt=cpt+1
                    fileout.write(str(pos_or_neg)+","+str(cptST)+","+str(cpt_ali)+","+str(cpt_pos)+","+str(Pro_1)+","+str(min1)+","+str(plus1)+","+str(plus2)+","+str(plus3)+","+str(plus4)+","+str(plus5)+","+str(score_flex_neg)+","+str(site_nature)+","+str(ss)+","+str(value_psh_psi)+"\n")
           
            i=i+1
fileout.close()
