#!/bin/bash
cd /home/user/wt_rep/code/active_learning_studies/pair_disjoint_not_image_disjoint
export PAIR_STUDY_SEEDS=500-534 JU_INITIAL=groups JU_MODE=both JU_THREADS=2
export JU_ONLY=random,random_pair_type,cluster_quota_uncertainty_lf,cluster_quota_uncertainty,ensemble_bald_decisive,ensemble_bald,fisher_dopt,bald_decisive,laplace_bald,core_set,core_set_relation,dpp_pairs,uncertainty,uncertainty_all_heads,delta_gap
SPLIT=$1
JU_SPLIT=$SPLIT JU_OUT=replication/${SPLIT}_groups python judgment_unit_study.py
