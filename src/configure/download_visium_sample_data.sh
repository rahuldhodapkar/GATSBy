## download_visium_sample_data.sh
#
# Visium is a commercial spatial transcriptomics platform provided
# by 10X Genomics. This script downloads the required files to a
# local data directory and checks the integrity of the download
# against the file versions which this software package was
# developed with.
#
# All scripts 
#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

set -e
set -x

########
# Normal Human Prostate
# https://www.10xgenomics.com/resources/datasets/normal-human-prostate-ffpe-1-standard-1-3-0
#
mkdir -p ./data/visium/normal_human_prostate

curl https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Normal_Prostate/Visium_FFPE_Human_Normal_Prostate_filtered_feature_bc_matrix.h5 \
    -o ./data/visium/normal_human_prostate/filtered_feature_bc_matrix.h5


curl https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Normal_Prostate/Visium_FFPE_Human_Normal_Prostate_filtered_feature_bc_matrix.tar.gz \
    -o ./data/visium/normal_human_prostate/filtered_feature_bc_matrix.tar.gz


curl https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Normal_Prostate/Visium_FFPE_Human_Normal_Prostate_spatial.tar.gz \
    -o ./data/visium/normal_human_prostate/spatial.tar.gz

# Check all files
md5sum -c ./checksums/visium_files.md5

echo "Downloaded Visium Sample Data Sets"
echo "All done!"
