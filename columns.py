numeric_cols = ['age_at_initial_pathologic']
cat_cols = ['CNCluster',
 'COCCluster',
 'MethylationCluster',
 'OncosignCluster',
 'RNASeqCluster',
 'RPPACluster',
 'ethnicity',
 'gender',
 'histological_type',
 'laterality',
 'miRNACluster',
 'neoplasm_histologic_grade',
 'race',
 'tumor_location',
]

mode_impute_cols = ['race', 'ethnicity']

minus_one_impute = ['OncosignCluster',
 'histological_type',
 'RPPACluster',
 'miRNACluster',
 'neoplasm_histologic_grade',
 'tumor_tissue_site',
 'MethylationCluster',
 'laterality',
 'gender',
 'RNASeqCluster',
 'CNCluster',
 'COCCluster',
 'tumor_location']

ordinal_cols = ['OncosignCluster',
                'miRNACluster',
                'RNASeqCluster',
                'COCCluster',
                'RPPACluster',
                'histological_type',
                'neoplasm_histologic_grade',
                'CNCluster',
                'MethylationCluster']

freq_cols = ['laterality',
             'gender',
             'race',
             'tumor_location',
             'ethnicity']

y = ['death01']


cols_to_drop = ['death01',
                'Patient',
                'laterality',
                'gender',
                'race',
                'tumor_location',
                'ethnicity'
                ]