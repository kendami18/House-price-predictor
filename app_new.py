import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Load pickle files ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("cat_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("cat_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

FEATURE_NAMES = list(scaler.feature_names_in_)   # exact 73-feature order

CAT_OPTIONS = {
    "MSZoning":      ["C (all)", "FV", "RH", "RL", "RM"],
    "Street":        ["Grvl", "Pave"],
    "LotShape":      ["IR1", "IR2", "IR3", "Reg"],
    "LandContour":   ["Bnk", "HLS", "Low", "Lvl"],
    "Utilities":     ["AllPub", "NoSeWa"],
    "LotConfig":     ["Corner", "CulDSac", "FR2", "FR3", "Inside"],
    "LandSlope":     ["Gtl", "Mod", "Sev"],
    "Neighborhood":  ["Blmngtn","Blueste","BrDale","BrkSide","ClearCr","CollgCr",
                      "Crawfor","Edwards","Gilbert","IDOTRR","MeadowV","Mitchel",
                      "NAmes","NPkVill","NWAmes","NoRidge","NridgHt","OldTown",
                      "SWISU","Sawyer","SawyerW","Somerst","StoneBr","Timber","Veenker"],
    "Condition1":    ["Artery","Feedr","Norm","PosA","PosN","RRAe","RRAn","RRNe","RRNn"],
    "Condition2":    ["Artery","Feedr","Norm","PosA","PosN","RRAe","RRAn","RRNn"],
    "BldgType":      ["1Fam","2fmCon","Duplex","TwnhsE","Twnhs"],
    "HouseStyle":    ["1.5Fin","1.5Unf","1Story","2.5Fin","2.5Unf","2Story","SFoyer","SLvl"],
    "RoofStyle":     ["Flat","Gable","Gambrel","Hip","Mansard","Shed"],
    "RoofMatl":      ["ClyTile","CompShg","Membran","Metal","Roll","Tar&Grv","WdShake","WdShngl"],
    "Exterior1st":   ["AsbShng","AsphShn","BrkComm","BrkFace","CBlock","CemntBd","HdBoard",
                      "ImStucc","MetalSd","Plywood","Stone","Stucco","VinylSd","Wd Sdng","WdShing"],
    "Exterior2nd":   ["AsbShng","AsphShn","Brk Cmn","BrkFace","CBlock","CmentBd","HdBoard",
                      "ImStucc","MetalSd","Other","Plywood","Stone","Stucco","VinylSd",
                      "Wd Sdng","Wd Shng"],
    "MasVnrType":    ["BrkCmn","BrkFace","None","Stone"],
    "ExterQual":     ["Ex","Fa","Gd","Po","TA"],
    "ExterCond":     ["Ex","Fa","Gd","Po","TA"],
    "Foundation":    ["BrkTil","CBlock","PConc","Slab","Stone","Wood"],
    "BsmtQual":      ["Ex","Fa","Gd","NA","Po","TA"],
    "BsmtCond":      ["Ex","Fa","Gd","NA","Po","TA"],
    "BsmtExposure":  ["Av","Gd","Mn","NA","No"],
    "BsmtFinType1":  ["ALQ","BLQ","GLQ","LwQ","NA","Rec","Unf"],
    "BsmtFinType2":  ["ALQ","BLQ","GLQ","LwQ","NA","Rec","Unf"],
    "Heating":       ["Floor","GasA","GasW","Grav","OthW","Wall"],
    "HeatingQC":     ["Ex","Fa","Gd","Po","TA"],
    "CentralAir":    ["N","Y"],
    "Electrical":    ["FuseA","FuseF","FuseP","Mix","SBrkr"],
    "KitchenQual":   ["Ex","Fa","Gd","Po","TA"],
    "Functional":    ["Maj1","Maj2","Min1","Min2","Mod","Sal","Sev","Typ"],
    "FireplaceQu":   ["Ex","Fa","Gd","NA","Po","TA"],
    "GarageType":    ["2Types","Attchd","Basment","BuiltIn","CarPort","Detchd","NA"],
    "GarageFinish":  ["Fin","NA","RFn","Unf"],
    "GarageQual":    ["Ex","Fa","Gd","NA","Po","TA"],
    "GarageCond":    ["Ex","Fa","Gd","NA","Po","TA"],
    "PavedDrive":    ["N","P","Y"],
    "SaleType":      ["COD","CWD","Con","ConLD","ConLI","ConLw","New","Oth","WD"],
    "SaleCondition": ["Abnorml","AdjLand","Alloca","Family","Normal","Partial"],
}

CAT_COLS = [f for f in FEATURE_NAMES if f in CAT_OPTIONS]

@st.cache_resource
def build_encoders():
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col, options in CAT_OPTIONS.items():
        le = LabelEncoder()
        le.fit(options)
        encoders[col] = le
    return encoders

encoders = build_encoders()

NUM_DEFAULTS = {
    "MSSubClass": 50, "LotFrontage": 69, "LotArea": 9478,
    "OverallQual": 6, "OverallCond": 5, "YearBuilt": 1973,
    "YearRemodAdd": 1994, "MasVnrArea": 0, "BsmtFinSF1": 383,
    "BsmtFinSF2": 0, "BsmtUnfSF": 477, "TotalBsmtSF": 991,
    "1stFlrSF": 1087, "2ndFlrSF": 0, "LowQualFinSF": 0,
    "GrLivArea": 1464, "BsmtFullBath": 0, "BsmtHalfBath": 0,
    "FullBath": 2, "HalfBath": 0, "BedroomAbvGr": 3,
    "KitchenAbvGr": 1, "TotRmsAbvGrd": 6, "Fireplaces": 1,
    "GarageYrBlt": 1979, "GarageCars": 2, "GarageArea": 480,
    "WoodDeckSF": 0, "OpenPorchSF": 25, "EnclosedPorch": 0,
    "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
    "MiscVal": 0, "MoSold": 6, "YrSold": 2008,
}
CAT_DEFAULTS = {col: opts[0] for col, opts in CAT_OPTIONS.items()}

def predict(user_inputs: dict) -> float:
    row = {}
    for feat in FEATURE_NAMES:
        if feat in CAT_COLS:
            val = user_inputs.get(feat, CAT_DEFAULTS[feat])
            row[feat] = encoders[feat].transform([val])[0]
        else:
            row[feat] = user_inputs.get(feat, NUM_DEFAULTS.get(feat, 0))
    df = pd.DataFrame([row])[FEATURE_NAMES]
    scaled = scaler.transform(df)
    return float(np.expm1(model.predict(scaled)[0]))

with st.sidebar:
    st.header("Property Details")

    st.subheader("⭐ Quality & Condition")
    overall_qual  = st.slider("Overall Quality (1–10)", 1, 10, NUM_DEFAULTS["OverallQual"])
    overall_cond  = st.slider("Overall Condition (1–10)", 1, 10, NUM_DEFAULTS["OverallCond"])
    exter_qual    = st.selectbox("Exterior Quality", CAT_OPTIONS["ExterQual"],
                                  index=CAT_OPTIONS["ExterQual"].index("TA"))
    kitchen_qual  = st.selectbox("Kitchen Quality", CAT_OPTIONS["KitchenQual"],
                                  index=CAT_OPTIONS["KitchenQual"].index("TA"))
    heating_qc    = st.selectbox("Heating Quality", CAT_OPTIONS["HeatingQC"],
                                  index=CAT_OPTIONS["HeatingQC"].index("Ex"))

    st.subheader("📐 Size")
    gr_liv_area   = st.number_input("Above-Ground Living Area (sq ft)", 300, 6000, NUM_DEFAULTS["GrLivArea"])
    total_bsmt    = st.number_input("Total Basement (sq ft)", 0, 3000, NUM_DEFAULTS["TotalBsmtSF"])
    first_flr     = st.number_input("1st Floor (sq ft)", 300, 4000, NUM_DEFAULTS["1stFlrSF"])
    second_flr    = st.number_input("2nd Floor (sq ft)", 0, 2500, NUM_DEFAULTS["2ndFlrSF"])
    lot_area      = st.number_input("Lot Area (sq ft)", 1000, 200000, NUM_DEFAULTS["LotArea"])

    st.subheader("🛏️ Rooms & Bathrooms")
    bedrooms      = st.number_input("Bedrooms Above Grade", 0, 10, NUM_DEFAULTS["BedroomAbvGr"])
    full_bath     = st.number_input("Full Bathrooms", 0, 5, NUM_DEFAULTS["FullBath"])
    half_bath     = st.number_input("Half Bathrooms", 0, 3, NUM_DEFAULTS["HalfBath"])
    tot_rms       = st.number_input("Total Rooms Above Grade", 2, 15, NUM_DEFAULTS["TotRmsAbvGrd"])
    fireplaces    = st.number_input("Fireplaces", 0, 4, NUM_DEFAULTS["Fireplaces"])

    st.subheader("🏗️ Build & Location")
    year_built    = st.number_input("Year Built", 1872, 2025, NUM_DEFAULTS["YearBuilt"])
    year_remod    = st.number_input("Year Remodeled", 1950, 2025, NUM_DEFAULTS["YearRemodAdd"])
    neighborhood  = st.selectbox("Neighborhood", CAT_OPTIONS["Neighborhood"],
                                  index=CAT_OPTIONS["Neighborhood"].index("NAmes"))
    house_style   = st.selectbox("House Style", CAT_OPTIONS["HouseStyle"],
                                  index=CAT_OPTIONS["HouseStyle"].index("1Story"))
    bldg_type     = st.selectbox("Building Type", CAT_OPTIONS["BldgType"],
                                  index=CAT_OPTIONS["BldgType"].index("1Fam"))
    foundation    = st.selectbox("Foundation", CAT_OPTIONS["Foundation"],
                                  index=CAT_OPTIONS["Foundation"].index("PConc"))
    central_air   = st.selectbox("Central Air", CAT_OPTIONS["CentralAir"],
                                  index=CAT_OPTIONS["CentralAir"].index("Y"))

    st.subheader("🚗 Garage")
    garage_cars   = st.number_input("Garage Car Capacity", 0, 5, NUM_DEFAULTS["GarageCars"])
    garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500, NUM_DEFAULTS["GarageArea"])
    garage_type   = st.selectbox("Garage Type", CAT_OPTIONS["GarageType"],
                                  index=CAT_OPTIONS["GarageType"].index("Attchd"))
    garage_finish = st.selectbox("Garage Finish", CAT_OPTIONS["GarageFinish"],
                                  index=CAT_OPTIONS["GarageFinish"].index("Unf"))
    garage_yr     = st.number_input("Garage Year Built", 1900, 2025, NUM_DEFAULTS["GarageYrBlt"])

    st.subheader("💰 Sale Info")
    sale_cond     = st.selectbox("Sale Condition", CAT_OPTIONS["SaleCondition"],
                                  index=CAT_OPTIONS["SaleCondition"].index("Normal"))
    sale_type     = st.selectbox("Sale Type", CAT_OPTIONS["SaleType"],
                                  index=CAT_OPTIONS["SaleType"].index("WD"))
    mo_sold       = st.slider("Month Sold", 1, 12, NUM_DEFAULTS["MoSold"])
    yr_sold       = st.selectbox("Year Sold", [2006, 2007, 2008, 2009, 2010],
                                  index=2)

    predict_btn = st.button("💰 Predict Sale Price", use_container_width=True, type="primary")

user_inputs = {
    "OverallQual":  overall_qual,  "OverallCond":  overall_cond,
    "ExterQual":    exter_qual,    "KitchenQual":  kitchen_qual,
    "HeatingQC":    heating_qc,    "GrLivArea":    gr_liv_area,
    "TotalBsmtSF":  total_bsmt,    "1stFlrSF":     first_flr,
    "2ndFlrSF":     second_flr,    "LotArea":      lot_area,
    "BedroomAbvGr": bedrooms,      "FullBath":     full_bath,
    "HalfBath":     half_bath,     "TotRmsAbvGrd": tot_rms,
    "Fireplaces":   fireplaces,    "YearBuilt":    year_built,
    "YearRemodAdd": year_remod,    "Neighborhood": neighborhood,
    "HouseStyle":   house_style,   "BldgType":     bldg_type,
    "Foundation":   foundation,    "CentralAir":   central_air,
    "GarageCars":   garage_cars,   "GarageArea":   garage_area,
    "GarageType":   garage_type,   "GarageFinish": garage_finish,
    "GarageYrBlt":  garage_yr,     "SaleCondition":sale_cond,
    "SaleType":     sale_type,     "MoSold":       mo_sold,
    "YrSold":       yr_sold,
}

#Main page
st.title("🏠 House Price Predictor")
st.caption("Powered by the CatBoostRegressor")

if predict_btn:
    price = predict(user_inputs)
    low, high = price * 0.90, price * 1.10

    res_col, gap_col = st.columns([2, 1])
    with res_col:
        st.success(f"## 💵 Predicted Sale Price")
        st.metric(label="Prediction", value=f"${price:,.0f}",
                  delta=f"Range: ${low:,.0f} – ${high:,.0f}")

    st.divider()
    st.subheader("📊 Top 15 Model Features by Importance")
    imp = pd.Series(model.get_feature_importance(), index=FEATURE_NAMES).nlargest(15).sort_values()
    st.bar_chart(pd.DataFrame({"Importance": imp}))

else:
    st.info("👈 Adjust the property details in the sidebar, then click **Predict Sale Price**.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pipeline")
        st.markdown("""
| Step | Detail |
|------|--------|
| Encoding | `LabelEncoder` per categorical column |
| Scaling  | `cat_scaler.pkl` for the `StandardScaler` |
| Model    | `cat_model.pkl` for the `CatBoostRegressor` |
        """)
    with col2:
        st.subheader("Model Info")
        st.markdown(f"""
| Property | Value |
|----------|-------|
| Features | {model.n_features_in_} |
| Iterations | {model.get_param('iterations') or 'default'} |
| Loss function | {model.get_param('loss_function') or 'RMSE'} |
        """)
