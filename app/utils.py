import pandas as pd
from flask import request


def extract_feature_values(data):
    """ Given a params dict, return the values for feeding into a model"""

    # Replace these features with the features for your model. They need to
    # correspond with the `name` attributes of the <input> tags
    EXPECTED_FEATURES = [
        'yr_adminst',
        'region',
        'msa',
        'evr_smoked_cig',
        'cig_month_freq',
        'alcohol_lifetime_freq',
        'alcohol_yr_freq',
        'alcohol_2weeks',
        'sex',
        'area_type',
        'marital_status',
        'has_father',
        'has_mother',
        'has_siblings',
        'father_educ_lvl',
        'mother_educ_lvl',
        'mother_employed',
        'political_value_type',
        'relig_attd',
        'relig_importance',
        'academic_self_rating',
        'intelligence_self_rating',
        'school_missed_illness',
        'school_missed_ditched',
        'school_missed_other',
        'skipped_class',
        'avg_grade',
        'tech_school_after_hs',
        'military_after_hs',
        '2yrcoll_after_hs',
        '4yrcoll_after_hs',
        'gradsch_after_hs',
        'desire_tech_school',
        'desire_military',
        'desire_2yrcoll',
        'desire_4yrcoll',
        'desire_gradsch',
        'desire_none',
        'work_hrs',
        'work_pay',
        'other_income',
        'rec_time',
        'date_freq',
        'drive_freq',
        '12mo_r_tcktd',
        '12mo_accidents']

    values = [[]]

    for feature in EXPECTED_FEATURES:
        resp = request.form.get(feature)
        values[0].append(resp)
        # if feature not in checkbox_feats:
        #     values[0].append(data[feature])
        # elif feature in checkbox_feats:
        #     if data[feature]!='Yes':
        #         values[0].append('No')
        #     else:
        #         values[0].append(data[feature])
    return pd.DataFrame(values, columns=EXPECTED_FEATURES)
