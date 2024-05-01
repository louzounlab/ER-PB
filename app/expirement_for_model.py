import pandas as pd

from os.path import join
import pickle

def process_data():
    data = pd.read_csv(
        join('static', 'all_inputs.csv'), skiprows=1)
    # Filling null values with median
    data.fillna(data.median(), inplace=True)
    # hospitalization parametes: "Fever_Min_during_hospitalization", "pulse_Min_during_hospitalization", "Saturation_Min_during_hospitalization", "Diastolic _BP_Min_during_hospitalization", "Systolic _BP_Min_during_hospitalization", "Cervical_length_Min_during_hospitalization", "Fever_Max_during_hospitalization", "pulse_max_during_hospitalization", "Saturation_Max_during_hospitalization", "Diastolic _BP_Max_during_hospitalization", "Systolic _BP_Max_during_hospitalization", "Cervical_length_Max_during_hospitalization", "Cervical_dilation_Max_during_hospitalization", "Cervical_effacement_Max_during_hospitalization", "Days with fever above 38", "Days of pulse above 100", "WBC_Min", "HB_Min", "PLT_Min", "ALK.PHOSPHATASE_Min", "AST_Min", "ALT_Min", "C-REACTIVE PROTEIN_Min", "MCV_Min", "RDW_Min", "Total Bile Acids_Min", "Transferrin_Min", "Albumin_Min", "Calcium_Min", "Protein-total_Min", "Phosphorus_Min", "TSH_Min", "T3-FREE_Min", "T4-FREE_Min", "Protein-Urine 24h_Min", "Creatinine- U sample_Min", "Protein - U sample_Min", "WBC_Max", "HB_Max", "PLT_Max", "ALK.PHOSPHATASE_Max", "AST_Max", "ALT_Max", "C-REACTIVE PROTEIN_Max", "CREATININE_Max", "MCV_Max", "RDW_Max", "GGT_Max", "Total Bile Acids_Max", "Transferrin_Max", "Ferritin_Max", "Albumin_Max", "Calcium_Max", "Protein-total_Max", "Phosphorus_Max", "TSH_Max", "T3-FREE_Max", "T4-FREE_Max", "Protein-Urine 24h_Max", "Creatinine- U sample_Max", "Protein - U sample_Max"
    return data


def safe_drop(data, columns=[], colrange=()):
    if colrange:
        if colrange[0] in data.columns and colrange[1] in data.columns:
            original_cols = list(data.columns.values)
            columns += [original_cols[i] for i in
                        range(original_cols.index(colrange[0]), original_cols.index(colrange[1]) + 1)]
    for col in columns:
        if col in data.columns:
            data.drop(col, inplace=True, axis=1)


def main():
    # all_data = process_data()
    #
    # # Create a dataframe with the exact same columns as the original dataframe and the medians as the values
    # median_data = pd.DataFrame(all_data.median()).T
    #
    # # Drop the columns that are not needed
    # safe_drop(median_data, columns=["Delivered within 2 days from admission", "Delivered before 34 yes no", "Delivered within 7 days from admission", "Delivered before 37 yes no"])
    #
    # # Save the median data to a csv file
    # median_data.to_csv(join("static", "median_data.csv"), index=False)
    #
    # with open(join("models", "xgboost_34.pkl"), "rb") as f:
    #     model = pickle.load(f)
    #
    # print(model.predict_proba(median_data))

    lst = [('gest_age', ''), ('gravidity', ''), ('parity', ''), ('prev_abo', ''), ('eup', '1'), ('prev_ces_del', ''), ('vbac', ''), ('living_children', ''), ('diab_mel', '1'), ('ges_hype_dis', '1'), ('hypothyroidism', '0'), ('hyperthyroidism', '0'), ('asthma', '0'), ('depression', '0'), ('anxiety', '0'), ('bipolar', '0'), ('epilepsy', '0'), ('anemia thalassemia', '0'), ('lupus', '0'), ('apla', '0'), ('other_rheumatoid', '0'), ('uc_or_crohn', '0'), ('max_pulse', ''), ('fetal_pe', '1'), ('gct', ''), ('number_of_fetuses', ''), ('prev_hos', '0'), ('pprom', '0'), ('premature_contractions', '0'), ('cervical_dynamics', '0'), ('placental_abruption', '0'), ('ogtt_fasting_glucose', ''), ('ogtt_h1', ''), ('ogtt_h2', ''), ('ogtt_h3', ''), ('nuchal_trans', ''), ('first_trimester_ds', ''), ('first_trimester_pappa', ''), ('first_trimester_hcg', ''), ('second_trimester_ds', ''), ('invasive_prenatal_testing', ''), ('ac', '0'), ('cvs', '0'), ('umbilical_cord_puncture', '0'), ('testing_type', ''), ('abnormal_genetic_testing_result', '0'), ('estimated_fetal_weight', ''), ('placental_location', ''), ('amniotic_fluid_index', ''), ('presentation_fetus_2', ''), ('fetus_2_vertex', 'Nan'), ('fetus_2_transverse_oblique', 'Nan'), ('amniotic_fluid_amount_fetus_2', ''), ('fetus_2_oligohydramnios', 'Nan'), ('fetus_2_polyhydramnios', 'Nan'), ('fetus_2_normal_afi', 'Nan'), ('cervical_dilation', ''), ('hemoglobin', ''), ('drugs', 'acid'), ('drugs', 'allergy'), ('drugs', 'anticonvulsants'), ('drugs', 'clexane'), ('selectedDrugs', 'acid;allergy;anticonvulsants;clexane'), ('microbes', 'gram_negative'), ('length', ''), ('weight', '')]
    lst = [a[0] for a in lst]

    print(lst)

if __name__ == "__main__":
    main()
