#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Variety calls from counts data using clustering
'''
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import umap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import graphs as plot
import randMatrix as rand
 
def processCounts(inFile, outFile):
    '''
    Process DArT tag counts file into the correct input format
    
    Args:
        inFile: name and path to counts file
        outFile: name and path for output file
    '''
    df = pd.read_csv(inFile, skiprows=7)
    notData = ['AlleleSequence','SNP','CallRate','OneRatioRef',
            'OneRatioSnp','FreqHomRef','FreqHomSnp','FreqHets',
            'PICRef','PICSnp','AvgPIC','AvgCountRef','AvgCountSnp',
            'RatioAvgCountRefAvgCountSnp']
    df = df.drop(columns=notData)
    df.to_csv(outFile, index=False)    
 
def processCountsSeq(inFile, outFile):  
    '''
    Process DArT seq counts file into the correct input format
    
    Args:
        inFile: name and path to counts file
        outFile: name and path for output file
    '''  
    df = pd.read_csv(inFile, skiprows =7)
    notData = ['AlleleID', 'AlleleSequence', 'TrimmedSequence',
           'Chrom_Eragrostis_CogeV3', 'ChromPosTag_Eragrostis_CogeV3',
           'ChromPosSnp_Eragrostis_CogeV3', 'AlnCnt_Eragrostis_CogeV3',
           'AlnEvalue_Eragrostis_CogeV3', 'Strand_Eragrostis_CogeV3', 'SNP',
           'SnpPosition', 'CallRate', 'OneRatioRef', 'OneRatioSnp', 'FreqHomRef',
           'FreqHomSnp', 'FreqHets', 'PICRef', 'PICSnp', 'AvgPIC', 'AvgCountRef',
           'AvgCountSnp', 'RepAvg']
    df = df.drop(columns=notData)    
    df.rename(columns={'CloneID': 'MarkerName'}, inplace=True)
    df.to_csv(outFile, index=False)    

def filterData(countsFile, metaFile, minloci, minSample, refFilter = None):
    '''
    Input the reformatted counts file and paired metadata file, filter out low quality samples/genes and then interpolate missing data 
    
    Args:
        countsFile: path to reformatted counts file
        metaFile: path to the metadata file paired with the countsFile
        refFilter: (optional) remove references with a divergence score above this value
    '''
    #import counts data
    counts = pd.read_csv(countsFile, index_col='MarkerName')
    snpProportion = counts.groupby(['MarkerName']).first()/counts.groupby(['MarkerName']).sum() #if the count for both SNPs are zero --> NaN
    
    #import sample metadata
    sampleMeta = pd.read_csv(metaFile)
    refRemove = sampleMeta[sampleMeta['reference'] == 'REMOVE']['short_name'].values.astype('str')
    
    #optionally remove references above a divergence cutoff
    if refFilter:
        divergent = snpProportion.columns[plot.homozygousDivergence(snpProportion) > refFilter].astype('int') #all samples above cutoff
        references = sampleMeta[(sampleMeta['reference'].notna())]['short_name'].values
        refRemove = np.append(refRemove, sampleMeta[sampleMeta['short_name'].isin(np.intersect1d(divergent, references))]['short_name'].values.astype('str'))
        
    snpProportion = snpProportion.drop(refRemove, axis=1)
    sampleMeta = sampleMeta.drop(sampleMeta[sampleMeta['short_name'].isin(refRemove.astype('int'))].index)
    
    snpProportionNoInterpolation = snpProportion.copy()
    
    #filter snpProportion for samples and genes with too many NaN
    snpProportion = snpProportion[snpProportion.isna().sum(axis = 1) < (minloci*snpProportion.shape[1])] #remove genes
    snpProportion = snpProportion.drop(columns=snpProportion.columns[snpProportion.isna().sum(axis = 0) > (minSample*snpProportion.shape[0])]) #remove samples

    #interpolate NaN using gene average
    snpProportion = snpProportion.where(pd.notna(snpProportion), snpProportion.mean(axis = 1), axis='rows')
    
    return snpProportion, snpProportionNoInterpolation, sampleMeta

def embedData(snpProportion, umapSeed):
    '''
    Input the processed snpProprtion data, embed with UMAP, and then cluster using DBSCAN
    
    Args:
        snpProportion: processed SNP proportion data
        umapSeed: RNG seed for umap embedding
    '''
    #UMAP embedding
    reducer = umap.UMAP(random_state=umapSeed)
    embedding = reducer.fit_transform(snpProportion.T) #the order is the same after embedding
    
    return embedding


def clusteringDBSCAN(snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff, outputDir):
    '''
    Input the processed snpProprtion data, embed with UMAP, and then cluster using DBSCAN
    
    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        embedding: UMAP embedding of snpProportion
        epsilon: epsilon parameter for DBSCAN clustering
        filePrefix: prefix for output filenames
        outputDir: directory for output files
    '''    
    #cluster using DBSCAN
    db_communities = DBSCAN(eps=epsilon, min_samples=2).fit(embedding).labels_
    
    #save output figures
    plot.umapCluster(embedding, db_communities)
    plt.savefig(getOutputFilepath(outputDir, filePrefix+' UMAP DBSCAN (epsilon ' + str(epsilon)+').png'), dpi = 300)
    
    plot.umapReference(snpProportion, embedding, sampleMeta, db_communities)
    plt.savefig(getOutputFilepath(outputDir, filePrefix+' UMAP references (DBSCAN clusters, epsilon ' + str(epsilon)+').png'), dpi = 300)

    if admixedCutoff:
        plot.histogramDivergence(snpProportion,sampleMeta)
        plt.savefig(getOutputFilepath(outputDir, filePrefix+' histogram divergence.png'), dpi = 300)
   
    return db_communities

def evaluateEpsilon(embedding, filePrefix, outputDir):
    '''
    Evaluate different epsilon values for DBSCAN
        
    Args:
        embedding: UMAP embedding of snpProportion
        filePrefix: prefix for output filenames
        outputDir: directory for output files
    '''
    ks = np.around(np.arange(0.1,1.1,0.05), 2) # Range of epsilon values for DBSCAN
    rand.randScoreMatrix(embedding, ks, 'DBSCAN')
    plt.savefig(getOutputFilepath(outputDir, filePrefix+' DBSCAN rand matrix.png'), dpi = 300)    

def evaluateCutHeight(snpProportion, sampleMeta, db_communities, admixedCutoff, minRepTogether = 0.0, maxVarietyTogether = 4):
    '''
    Evaluate different cut height values for processing the dendrogram
    
    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        db_communities: DBSCAN cluster number for each sample
        admixedCutoff: clades without a reference and a minimum divergence value above this will be labeled as admixed
        minRepTogether: the minimum proportion of reference technical replicates that are in the same clade
        maxVarietyTogether: the maximum average number of varieties in the same clade (for clusters with at least one reference)
    '''

    #evaluate using the largest cluster 
    db_cluster, db_counts = np.unique(db_communities, return_counts=True)
    mainCluster = db_cluster[np.where(db_counts == max(db_counts))[0][0]]
    clusterSubsetLarge = snpProportion[snpProportion.columns[np.where(db_communities == mainCluster)]]
    Y_clusterLarge = sch.linkage(clusterSubsetLarge.values.T, metric='correlation') #sort samples
    rep, avg, totalRef, cuts = rand.cutoffQuality(clusterSubsetLarge, sampleMeta, Y_clusterLarge)

    ks = np.around(np.intersect1d(cuts[np.where(rep > minRepTogether*totalRef)], cuts[np.where(avg < maxVarietyTogether)]),3) 
    rand.randScoreMatrix(snpProportion, ks, 'HC', sampleMeta = sampleMeta, admixedCutoff = admixedCutoff)

def labelSamples(snpProportion,sampleMeta,db_communities,embedding, cutHeight, admixedCutoff, filePrefix, outputDir):
    '''
    Evaluate different cut height values for processing the dendrogram
    
    Args:
        snpProportion: processed SNP proportion data
        sampleMeta: metadata paired with genotyping data
        db_communities: DBSCAN cluster number for each sample
        embedding: UMAP embedding of snpProportion
        cutHeight: cutoff value for cutting a dendrogram into clusters
        admixedCutoff: clades without a reference and a minimum divergence value above this will be labeled as admixed
        filePrefix: prefix for output filenames
        outputDir: directory for output files
    '''
    #consolidate outputs
    output = pd.DataFrame(embedding, columns=['embedding_X', 'embedding_Y'])
    output['cluster'] = db_communities
    output['short_name'] = snpProportion.columns
    if admixedCutoff:
        output['divergence'] = plot.homozygousDivergence(snpProportion)
    output['variety'] = pd.NA
    
    #for each short_name, find the index in sampleMeta and grab the alt_name
    countryID = []
    for short_name in snpProportion.columns:
        countryID.append(sampleMeta[sampleMeta['short_name'] == int(short_name)]['alt_name'].values[0])
    output['countryID'] = countryID
    
    
    for cluster in np.unique(db_communities):
        if cluster == -1: #-1 indicates samples that are disconnected from the rest of the clusters  
            subsetIndex = np.where(db_communities == cluster)[0]
            output.loc[subsetIndex,'variety'] = 'Admixed'
            
        else:
            #subset for a single DBSCAN cluster
            subsetIndex = np.where(db_communities == cluster)[0]
                
            #cluster subset of samples using heirarchical clustering
            Y_cluster = sch.linkage(snpProportion[snpProportion.columns[subsetIndex]].values.T, metric='correlation')
            
            #label samples
            communities, names = rand.labelHCLandrace(snpProportion[snpProportion.columns[subsetIndex]], sampleMeta, Y_cluster, cutHeight, clusterNumber = cluster, admixedCutoff = admixedCutoff)
            varietiesList = []
            for i in communities.astype('int'):
                varietiesList.append(names[i][0])          
            output.loc[subsetIndex,'variety'] = varietiesList
    
    #save outputs
    plot.umapRefLandrace(snpProportion, output, sampleMeta, 5, noRef=True)
    plt.savefig(getOutputFilepath(outputDir, filePrefix+' UMAP clustering predictions (cut height'+str(cutHeight)+').png'), dpi = 300)
    
    plot.barchartRef(snpProportion, output, sampleMeta)
    plt.savefig(getOutputFilepath(outputDir, filePrefix+' bar chart clustering predictions (cut height'+str(cutHeight)+').png'), dpi = 300)
    
    output.to_csv(getOutputFilepath(outputDir, filePrefix+'_clusteringOutputData_cutHeight'+str(cutHeight)+'.csv'), index=False)
    
    return output

def getOutputFilepath(outputDir, filename):
    return os.path.join(outputDir, os.path.basename(filename))

def main():
    if not parameters:
        print("""No parameters provided. Please provide parameters via: 
        -j or --json CLI argument passing in a JSON string 
        -f or --file CLI argument passing in a JSON file path 
        or set the parametersFile variable to the path of a JSON file in your environment""")
        raise Exception('No parameters provided')

    '''
    Args:
        outputDir: directory for output files, defaults to ./output
        minSample: samples must be missing from less than X loci
        minloci: loci must be absent from less than Y samples
        umapSeed: RNG seed for UMAP
        epsilon: epsilon value for DBSCAN
        cutHeight: cutoff value for cutting a dendrogram into clusters
        admixedCutoff: clades without a reference and a minimum divergence value above this will be labeled as admixed, null will be interpreted by JSON as None
        filePrefix: prefix for output filenames
        inputCountsFile: name and path to DArT counts file
        inputMetaFile: name and path to the metadata file paired with the countsFile
    ''' 
    outputDir = parameters.get('outputDir', './output')
    minSample = parameters["minSample"]
    minloci = parameters["minloci"]
    umapSeed = parameters["umapSeed"]
    epsilon = parameters["epsilon"]
    cutHeight = parameters["cutHeight"]
    admixedCutoff = parameters["admixedCutoff"]
    filePrefix = parameters["filePrefix"]
    inputCountsFile = parameters["inputCountsFile"]
    inputMetaFile = parameters["inputMetaFile"]

    # Create output directory if it doesn't exist
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    snpProportion, snpProportionNoInterpolation, sampleMeta = filterData(inputCountsFile, inputMetaFile, minloci, minSample)
    embedding = embedData(snpProportion, umapSeed)
    db_communities = clusteringDBSCAN(snpProportion, sampleMeta, embedding, epsilon, filePrefix, admixedCutoff, outputDir)
    output = labelSamples(snpProportion, sampleMeta, db_communities, embedding, cutHeight, admixedCutoff, filePrefix, outputDir)


parameters = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variety calls from counts data using clustering')
    parser.add_argument('-f', '--file', help='JSON file path with parameters', required=False)
    parser.add_argument('-j', '--json', type=json.loads, help='JSON string with parameters', required=False)
    args = parser.parse_args()

    if args.json:
        parameters = args.json
    elif args.file:
        parameters = json.loads(open(args.file).read())

if not parameters and ('parameterFile' in locals() or 'parameterFile' in globals()):
    parameters = json.loads(open(parameterFile).read())

main()
