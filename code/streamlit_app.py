import streamlit as st
from streamlit_utils import (
    abc,
    get_ccf_data,
    version,
    has_realigned_asset,
    ss_to_qp,
    ss_from_qp,
)
from abc_merfish_analysis import ccf_plots as cplots

FIG3_STATE = "/?extend_borders_qp=False&mg_focus_qp=True&bn_group_qp=None&de_coreonly_qp=False&bn_borders_qp=False&bs_tax_qp=subclass&bs_leg_qp=False&bn_coreonly_qp=False&gp_focus_qp=True&de_tax_qp=cluster&bn_anno_qp=automated&realigned_qp=False&transform_qp=log2cpt&de_data_qp=WMB-10Xv3&bn_tax_qp=cluster&mg_dark_qp=True&devccf_qp=False&de_anno_qp=automated&gp_regionlist_qp=AM&gp_gene_qp=Pdyn&gp_sectionlist_qp=C57BL6J-638850.44&mg_regionlist_qp=AM&mg_sectionlist_qp=C57BL6J-638850.44&mg_genelist_qp=Pdyn&mg_genelist_qp=Shisa6&mg_genelist_qp=Cacng3&bn_typelist_qp=2676+TH+Prkcd+Grin2c+Glut_9&bn_typelist_qp=2675+TH+Prkcd+Grin2c+Glut_9&bn_typelist_qp=2674+TH+Prkcd+Grin2c+Glut_9&bn_typelist_qp=2668+TH+Prkcd+Grin2c+Glut_8&bn_regionlist_qp=AM"
FIG4_STATE = "/?extend_borders_qp=False&mg_focus_qp=True&bn_group_qp=None&de_coreonly_qp=False&bn_borders_qp=False&bs_tax_qp=subclass&bs_leg_qp=False&bn_coreonly_qp=False&gp_focus_qp=True&de_tax_qp=cluster&bn_anno_qp=automated&realigned_qp=False&transform_qp=log2cpt&de_data_qp=WMB-10Xv3&bn_tax_qp=cluster&mg_dark_qp=True&devccf_qp=False&de_anno_qp=automated&gp_regionlist_qp=VPM&gp_regionlist_qp=VPL&gp_regionlist_qp=PO&gp_gene_qp=Kcnab3&gp_sectionlist_qp=C57BL6J-638850.37&mg_regionlist_qp=VPM&mg_regionlist_qp=VPL&mg_regionlist_qp=PO&mg_sectionlist_qp=C57BL6J-638850.37&mg_genelist_qp=Scn4b&mg_genelist_qp=Kcnab3&mg_genelist_qp=Calb1&bn_typelist_qp=2663+TH+Prkcd+Grin2c+Glut_6&bn_typelist_qp=2648+TH+Prkcd+Grin2c+Glut_1&bn_typelist_qp=2649+TH+Prkcd+Grin2c+Glut_1&bn_regionlist_qp=VPL&bn_regionlist_qp=VPM&bn_regionlist_qp=PO"

ss_to_qp()
ss_from_qp()
ss = st.session_state
pg = st.navigation([
    st.Page("streamlit_main.py", default=True), 
    # st.Page("streamlit_annotation.py")
])
st.set_page_config(page_title="Thalamus MERFISH explorer", layout="wide")

@st.dialog("THALMANAC help")
def help_popup(message):
    st.write(message)

if not ss.keys():
    help_popup(
        f"""
        ## Example configurations:
        Follow these links to access states of the app pre-configured for particular explorations

        - [AM nucleus and related genes - Fig. 3]({st.context.url + FIG3_STATE})
        - [Somatosensory nuclei and related genes - Fig. 4]({st.context.url + FIG4_STATE})
        """
    )

with st.expander("CCF alignment settings"):
    realigned = st.radio(
        "CCF alignment",
        [False, True],
        index=0,
        key="realigned_qp",
        format_func=lambda realigned: "thalamus-specific section-wise affine alignment"
        if realigned
        else "published nonlinear alignment",
    )
    extend_borders = st.checkbox("Extend CCF borders", key="extend_borders_qp")
    devccf = st.checkbox("Use DevCCF (Paxinos-based) parcellation", key="devccf_qp")

if realigned and not has_realigned_asset:
    realigned = False
    st.warning("Realigned metadata not found, using published alignment")

cplots.CCF_REGIONS_DEFAULT = abc.get_thalamus_names(level='devccf' if devccf else None)
# always use CCFv3 names for user options
ss.th_subregion_names = abc.get_thalamus_names('structure', include_unassigned=False)
ccf_images, ccf_boundaries = get_ccf_data(realigned, devccf=devccf, lump_structures=True)
coords = "section" if realigned else "reconstructed"
ss.common_args = dict(
    x_col="x_" + coords,
    y_col="y_" + coords,
    boundary_img=ccf_boundaries,
    ccf_images=ccf_images,
    ccf_level='devccf' if devccf else 'structure',
)

with st.sidebar:
    st.markdown("## Dataset Details")
    st.write(f"Version: {version}")
    st.write(f"Gene data version: {abc.get_file_version('adata_raw')}")
    st.write(f"Metadata version: {abc.get_file_version('cell_metadata')}")
    st.write(f"CCF version: {abc.get_file_version('ccf_metadata')}")
    st.write(f"Taxonomy ID: {abc.taxonomy_id}")

# extend size of multiselects for full section names
st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 280px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

pg.run()
