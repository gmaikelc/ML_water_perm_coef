# -*- coding: utf-8 -*-
"""
Created on Fri March 14 17:41:37 2024

@author: Gerardo Casanola
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from rdkit import Chem
from openbabel import openbabel
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go

# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64


#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='ml H2O permeability coefficient predictor', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

st.write("[![Website](https://img.shields.io/badge/website-RasulevGroup-blue)](http://www.rasulev.org)")
st.subheader("📌" "About Us")
st.markdown("The group of Prof. Rasulev is focused on development of artificial intelligence (AI)-based predictive models to design novel polymeric materials, nanomaterials and to predict their various properties, including toxicity, solubility, fouling release properties, elasticity, degradation rate, biodegradation, etc. The group applies computational chemistry, machine learning and cheminformatics methods for modeling, data analysis and development of predictive structure-property relationship models to find structural factors responsible for activity of investigated materials.")


# Introduction
#---------------------------------#

st.title(':computer: _ml_H2O water permeant coeff predictor_ ')

st.write("""

**It is a free web-application for Glass Transition Temperature Prediction**

The glass transition temperature (Tg) is one of the most important properties of polymeric materials and indicates an approximate temperature below which a macromolecular system changes from a relatively soft, 
flexible and rubbery state to a hard, brittle and glass-like one1. The Tg value also determines the utilization limits of many rubbers and thermoplastic materials. 
Besides, the drastic changes in the mobility of the molecules in different glassy states (from the frozen to the thawed state) affect many other chemical and physical properties, 
such as mechanical modulus, acoustical properties, specific heat, viscosity, mechanical energy absorption, density, dielectric coefficients, viscosity and the gases and liquids difussion rate in the polymer material. 
The change of these mechanical properties also specifies the employment of the material and the manufacturing process.

The Tg ML predictor is a Web App that use a SVM regression model to predict the glass transition temperature. 

The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [MOLVS](https://molvs.readthedocs.io/), [Openbabel](https://github.com/openbabel/openbabel),
[Scikit-learn](https://scikit-learn.org/stable/)
**Workflow:**
""")


image = Image.open('workflow_TGapp.png')
st.image(image, caption='Tg ML predictor workflow')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV file')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/gmaikelc/ML_water_perm_coef/main/example_file.csv) 
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a CSV file with SMILES and fractions", type=["csv"])




#%% Standarization by MOLVS ####
####---------------------------------------------------------------------------####

def standardizer(df,pos):
    s = Standardizer()
    molecules = df[pos].tolist()
    standardized_molecules = []
    smi_pos=pos-2
    i = 1
    t = st.empty()
    
    

    for molecule in molecules:
        try:
            smiles = molecule.strip()
            mol = Chem.MolFromSmiles(smiles)
            standarized_mol = s.super_parent(mol) 
            standardizer_smiles = Chem.MolToSmiles(standarized_mol)
            standardized_molecules.append(standardizer_smiles)
            # st.write(f'\rProcessing molecule {i}/{len(molecules)}', end='', flush=True)
            t.markdown("Processing monomers: " + str(i) +"/" + str(len(molecules)))

            i = i + 1
        except:
            standardized_molecules.append(molecule)
    df['standarized_SMILES'] = standardized_molecules

    return df


#%% Protonation state at pH 7.4 ####
####---------------------------------------------------------------------------####

def charges_ph(molecule, ph):

    # obConversion it's neccesary for saving the objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    
    # create the OBMol object and read the SMILE
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule)
    
    # Add H, correct pH and add H again, it's the only way it works
    mol.AddHydrogens()
    mol.CorrectForPH(7.4)
    mol.AddHydrogens()
    
    # transforms the OBMOl objecto to string (SMILES)
    optimized = obConversion.WriteString(mol)
    
    return optimized

def smile_obabel_corrector(smiles_ionized):
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize = False)
    
    # checks if the ether group is wrongly protonated
    pattern1 = Chem.MolFromSmarts('[#6]-[#8-]-[#6]')
    if mol1.HasSubstructMatch(pattern1):
        # gets the atom number for the O wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern1)
        at_matches_list = [y[1] for y in at_matches]
        # changes the charged for each O atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    pattern12 = Chem.MolFromSmarts('[#6]-[#8-]-[#16]')
    if mol1.HasSubstructMatch(pattern12):
        # gets the atom number for the O wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern12)
        at_matches_list = [y[1] for y in at_matches]
        # changes the charged for each O atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
            
    # checks if the nitro group is wrongly protonated in the oxygen
    pattern2 = Chem.MolFromSmarts('[#6][O-]=[N+](=O)[O-]')
    if mol1.HasSubstructMatch(pattern2):
        # print('NO 20')
        patt = Chem.MolFromSmiles('[O-]=[N+](=O)[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('O[N+]([O-])=O')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated in the oxygen
    pattern21 = Chem.MolFromSmarts('[#6]-[O-][N+](=O)=[O-]')
    if mol1.HasSubstructMatch(pattern21):
        # print('NO 21')
        patt = Chem.MolFromSmiles('[O-][N+](=O)=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[O][N+](=O)-[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]
        
    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern22 = Chem.MolFromSmarts('[#8-][N+](=[#6])=[O-]')
    if mol1.HasSubstructMatch(pattern22):
        # print('NO 22')
        patt = Chem.MolFromSmiles('[N+]([O-])=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[N+]([O-])-[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern23 = Chem.MolFromSmarts('[#6][N+]([#6])([#8-])=[O-]')
    if mol1.HasSubstructMatch(pattern23):
        # print('NO 23')
        patt = Chem.MolFromSmiles('[N+]([O-])=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[N+]([O-])[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern24 = Chem.MolFromSmarts('[#6]-[#8][N+](=O)=[O-]')
    if mol1.HasSubstructMatch(pattern24):
        # print('NO 24')
        patt = Chem.MolFromSmiles('[O][N+](=O)=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[O][N+](=O)[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the 1H-tetrazole group is wrongly protonated
    pattern3 = Chem.MolFromSmarts('[#7]-1-[#6]=[#7-]-[#7]=[#7]-1')
    if mol1.HasSubstructMatch(pattern3):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern3)
        at_matches_list = [y[2] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    # checks if the 2H-tetrazole group is wrongly protonated
    pattern4 = Chem.MolFromSmarts('[#7]-1-[#7]=[#6]-[#7-]=[#7]-1')
    if mol1.HasSubstructMatch(pattern4):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern4)
        at_matches_list = [y[3] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
        
    # checks if the 2H-tetrazole group is wrongly protonated, different disposition of atoms
    pattern5 = Chem.MolFromSmarts('[#7]-1-[#7]=[#7]-[#6]=[#7-]-1')
    if mol1.HasSubstructMatch(pattern5):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern4)
        at_matches_list = [y[4] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    smile_checked = Chem.MolToSmiles(mol1)
    return smile_checked



#%% formal charge calculation

def formal_charge_calculation(descriptors):
    smiles_list = descriptors["Smiles_OK"]
    charges = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            charge = Chem.rdmolops.GetFormalCharge(mol)
            charges.append(charge)
        except:
            charges.append(None)
        
    descriptors["Formal_charge"] = charges
    return descriptors


#%% Calculating molecular descriptors
### ----------------------- ###

def calc_descriptors(data, smiles_col_pos):
    descriptors_total_list = []
    smiles_list = []
    
    # Loop through each molecule in the dataset
    for pos, row in data.iterrows():
        molecule_name = row[0]  # Assuming the first column contains the molecule names
        molecule_smiles = row[smiles_col_pos]  # Assuming the specified column contains the SMILES
        
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            if mol is not None:
                smiles_ionized = charges_ph(molecule_smiles, 7.4)
                smile_checked = smile_obabel_corrector(smiles_ionized)
                smile_final = smile_checked.rstrip()
                smiles_list.append(smile_final)
                
                calc = Calculator(descriptors, ignore_3D=True)
                descriptor_values = calc(mol).asdict()
                
                # Create a dictionary with molecule name as key and descriptor values as values
                descriptors_dict = {'NAME': molecule_name}
                descriptors_dict.update(descriptor_values)
                
                descriptors_total_list.append(descriptors_dict)
        except:
            st.write(f'Molecule {molecule_name} has been removed (molecule not allowed by Mordred descriptor)')
    else:
        pass
    
    # Convert the list of dictionaries to a DataFrame
    descriptors_total = pd.DataFrame(descriptors_total_list)
    descriptors_total = descriptors_total.set_index('NAME',inplace=False).copy()
    descriptors_total = descriptors_total.reindex(sorted(descriptors_total.columns), axis=1)   
    descriptors_total.replace([np.inf, -np.inf], np.nan, inplace=True)
    descriptors_total["Smiles_OK"] = smiles_list
    
    # Perform formal charge calculation
    descriptors_total = formal_charge_calculation(descriptors_total)
    
    return descriptors_total, smiles_list


#%% normalizing data
### ----------------------- ###

def normalize_data(train_data, test_data):
    # Normalize the training data
    df_train = pd.DataFrame(train_data)
    saved_cols = df_train.columns
    min_max_scaler = preprocessing.MinMaxScaler().fit(df_train)
    np_train_scaled = min_max_scaler.transform(df_train)
    df_train_normalized = pd.DataFrame(np_train_scaled, columns=saved_cols)

    # Normalize the test data using the scaler fitted on training data
    np_test_scaled = min_max_scaler.transform(test_data)
    df_test_normalized = pd.DataFrame(np_test_scaled, columns=saved_cols)

    return df_train_normalized, df_test_normalized


#%% Determining Applicability Domain (AD)

def applicability_domain(x_test_normalized, x_train_normalized):
    
    X_train = x_train_normalized.values
    X_test = x_test_normalized.values
    # Calculate leverage and standard deviation for the training set
    hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverage_train = np.diagonal(hat_matrix_train)
    leverage_train=leverage_train.ravel()
    
    # Calculate leverage and standard deviation for the test set
    hat_matrix_test = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_test.T
    leverage_test = np.diagonal(hat_matrix_test)
    leverage_test=leverage_test.ravel()
    
    # threshold for the applicability domain
    
    h3 = 3*((x_train_normalized.shape[1]+1)/x_train_normalized.shape[0])  
    
    diagonal_compare = list(leverage_test)
    h_results =[]
    for valor in diagonal_compare:
        if valor < h3:
            h_results.append(True)
        else:
            h_results.append(False)         
    return h_results

#%% Removing molecules with na in any descriptor

def all_correct_model(descriptors_total,loaded_desc, smiles_list):
    
    total_desc = []
    for descriptor_set in loaded_desc:
        for desc in descriptor_set:
            if not desc in total_desc:
                total_desc.append(desc)
            else:
                pass
            
    X_final = descriptors_total[total_desc]
    X_final["SMILES_OK"] = smiles_list
    rows_with_na = X_final[X_final.isna().any(axis=1)]         # Find rows with NaN values
    for molecule in rows_with_na["SMILES_OK"]:
        st.write(f'\rMolecule {molecule} has been removed (NA value  in any of the necessary descriptors)')
    X_final1 = X_final.dropna(axis=0,how="any",inplace=False)
    
    smiles_final = X_final1["SMILES_OK"]
    return X_final1, smiles_final

 # Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Substrate":
        return 'lightgreen'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        return 'red'


#%% Predictions        

def predictions(loaded_model, loaded_desc, X_final1):
    scores = []
    palancas2 = []
    palancas3 = []

    i = 0
    
    for estimator in loaded_model:
        descriptors_model = loaded_desc[i]
        
        X = X_final1[descriptors_model]
        predictions = estimator.predict(X)
    
        scores.append(predictions)
        resultado_palanca2, resultado_palanca3  = applicability_domain(X, descriptors_model)
        palancas2.append(resultado_palanca2)
        palancas3.append(resultado_palanca3)
        i = i + 1 
    
    dataframe_scores = pd.DataFrame(scores).T
    dataframe_scores.index = smiles_final
    
    palancas_final2 = pd.DataFrame(palancas2).T
    palancas_final2.index = smiles_final
    palancas_final2['Confidence'] = (palancas_final2.sum(axis=1) / len(palancas_final2.columns)) * 100
    
    palancas_final3 = pd.DataFrame(palancas3).T
    palancas_final3.index = smiles_final
    palancas_final3['Confidence3'] = (palancas_final3.sum(axis=1) / len(palancas_final3.columns)) * 100

    score_ensemble = dataframe_scores.min(axis=1)
    classification = score_ensemble >= 0.44
    classification = classification.replace({True: 'Substrate', False: 'Non Substrate'})
    
    final_file = pd.concat([classification,palancas_final2['Confidence'], palancas_final3['Confidence3']], axis=1)
    
    final_file.rename(columns={0: "Prediction"},inplace=True)
    
    final_file.loc[final_file["Confidence"] >= 50, 'Confidence'] = 'HIGH'
    final_file.loc[(final_file["Confidence3"] >= 50) & (final_file["Confidence"] != "HIGH"), 'Confidence'] = 'MEDIUM'
    final_file.loc[final_file["Confidence3"] < 50, 'Confidence'] = 'LOW'

    final_file.loc[final_file["Confidence3"] < 50, 'Prediction'] = 'No conclusive'
    final_file.drop(columns=['Confidence3'],inplace=True)
            
    df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
    styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
    return final_file, styled_df



#%% Create plot:




def final_plot(final_file):
    non_conclusives = len(final_file[final_file['Confidence'] == "LOW"]) 
    substrates_hc = len(final_file[(final_file['Confidence'] == "HIGH") & (final_file['Prediction'] == 'Substrate')])
    substrates_mc = len(final_file[(final_file['Confidence'] == "MEDIUM") & (final_file['Prediction'] == 'Substrate')])

    # Count values in 'DA' column higher than 50 and 'class' is 'no'
    non_substrates_hc = len(final_file[(final_file['Confidence'] == "HIGH") & (final_file['Prediction'] == 'Non Substrate')])
    non_substrates_mc = len(final_file[(final_file['Confidence'] == "MEDIUM") & (final_file['Prediction'] == 'Non Substrate')])
    keys = ["Substrate - High confidence", "Substrate - Medium confidence", "Non Substrate - High confidence", "Non Substrate - Medium confidence", "Non conclusive"]
    fig = go.Figure(go.Pie(labels=keys, values=[substrates_hc, substrates_mc, non_substrates_hc, non_substrates_mc, non_conclusives]))
    
    #fig.update_layout(plot_bgcolor = 'rgb(256,256,256)', title_text="Global Emissions 1990-2011",
                            #title_font = dict(size=25, family='Calibri', color='black'),
                            #font =dict(size=20, family='Calibri'),
                            #legend_title_font = dict(size=18, family='Calibri', color='black'),
                            #legend_font = dict(size=15, family='Calibri', color='black'))
    
    fig.update_layout(title_text=None)
    
    return fig


#%%
def filedownload1(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="OCT1_class_results.csv">Download CSV File with results</a>'
    return href

#%% CORRIDA

loaded_model = pickle.load(open("models/" + "OCT1_models.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "OCT1_descriptors.pickle", 'rb'))


if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        data = pd.read_csv(uploaded_file_1,sep="\t",header=None)       
        descriptors_total, smiles_list = calcular_descriptores(data)
        X_final1, smiles_final = all_correct_model(descriptors_total,loaded_desc, smiles_list)
        final_file, styled_df = predictions(loaded_model, loaded_desc, X_final1)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)

        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)
       

# Example file
else:
    st.info('👈🏼👈🏼👈🏼      Awaiting for TXT file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        data = pd.read_csv("example_file.txt",sep="\t",header=None)
        descriptors_total, smiles_list = calcular_descriptores(data)
        X_final1, smiles_final = all_correct_model(descriptors_total,loaded_desc, smiles_list)
        final_file, styled_df = predictions(loaded_model, loaded_desc, X_final1)
        figure  = final_plot(final_file)  
        col1, col2 = st.columns(2)
        with col1:
            st.header("Predictions")
            st.write(styled_df)
        with col2:
            st.header("Resume")
            st.plotly_chart(figure,use_container_width=True)
  
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed with ❤️ by <a style='display: ;
 text-align: center' href="https://twitter.com/maxifallico" target="_blank">Maximiliano Fallico</a> ,  <a style='display:; 
 text-align: center' href="https://twitter.com/capigol" target="_blank">Lucas Alberca</a> and <a style='display: ; 
 text-align: center' href="https://twitter.com/carobellera" target="_blank">Caro Bellera</a> for <a style='display: ; 
 text-align: center;' href="https://lideb.biol.unlp.edu.ar/" target="_blank">LIDeB</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

