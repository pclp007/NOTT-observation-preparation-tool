import streamlit as st
import os
import scifysim as sf
import numpy as np
import matplotlib.pyplot as plt
import astropy.units
import pandas as pd
from scifysim.dummy import makesim
from astroplan import FixedTarget
from astropy.time import Time, TimeDelta
from astroplan import Observer
from pytz import timezone
from astroplan.plots import plot_sky
from astroplan.plots import plot_airmass
import datetime
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from cmcrameri import cm
import plotly
from astropy import units
from astroplan.plots import plot_finder_image
import dask



# Disable the submit button after it is clicked 
#def disable():
   #st.session_state.disabled = True


st.set_page_config(
     page_title='NOTT Observation Tool',
     layout="wide",
     initial_sidebar_state="expanded",
)

#st.sidebar.subheader('NOTT Observation Tool')

link = "https://www.aanda.org/articles/aa/full_html/2023/03/aa44351-22/aa44351-22.html"

st.sidebar.markdown('This is an observation preparation tool for the VLTI UTs/ATs, whose packages include: [SCIFYsim](%s) (the engine behind the tool), astropy & astroplan.'% link)  

with st.sidebar:
   # Initialize disabled for form_submit_button to False
   #if "disabled" not in st.session_state:
    #st.session_state.disabled = False

   with st.form("my_form"):
    option = st.selectbox( "which VLTI configuration would you like to use?", ("UT (UT1-UT2-UT3-UT4)", 
                                                                              "AT-large (AO-G1-J2-K0)", 
                                                                              "AT-medium (K0-G2-D0-J3)",
                                                                              #"AT-extended (AO-B5-J2-J6)" 
                                                                              "AT-small (A0-B2-D0-C1)"),
                                                                              index=None, placeholder="Select VLTI configuration...",)   
      
    # Assign a key to the widget so it's automatically in session state
    star_name = st.text_input('enter SIMBAD identifier of target', key="star_name")

    start_date = st.date_input('enter observation date', value=None)

    planet_ra = st.number_input('planet RA Offset (mas)', min_value=0)
    planet_dec = st.number_input('planet Dec Offset (mas)', min_value=0)

    submit = st.form_submit_button("submit")#, on_click=disable, disabled=st.session_state.disabled)

st.sidebar.markdown("The development of this tool is funded by the European Union's Horizon 2020 research and innovation program under grant agreement No. 101004719.")

st.subheader('NOTT observation tool')
tab1, tab2, tab3 = st.tabs(["general information", "throughput maps", "SNR maps"])


if submit:
   
   with tab1:
      col1, col2, col3, col4 = st.columns((1,1,1,1))

      target_1 = FixedTarget.from_name(star_name)


   ###################### DAILY CHART   ###########################
        
      col1.subheader('daily chart')

      from astropy.visualization import astropy_mpl_style, quantity_support

      plt.style.use(astropy_mpl_style)
      quantity_support()

      start_time = datetime.time(00, 00)
      obs_datetime = datetime.datetime.combine(start_date, start_time)
      observe_time = Time(obs_datetime)

      vlti = EarthLocation.of_site('Paranal Observatory (ESO)')
      target = SkyCoord.from_name(star_name)
      vlti_1 = Observer.at_site('Paranal Observatory (ESO)')
      utcoffset = -3 * u.hour  # Chile Summer Time (CLST)
      midnight = Time(obs_datetime) - utcoffset

      from astropy.coordinates import get_sun

      delta_midnight_sun = np.linspace(-12, 12, 1000) * u.hour
      times = midnight + delta_midnight_sun
      frame_sun = AltAz(obstime=times, location=vlti)
      sunaltazs = get_sun(times).transform_to(frame_sun)
 
      from astropy.coordinates import get_body
      from mpl_toolkits.axes_grid1 import make_axes_locatable

      labels = ['12:00','16:00','20:00','00:00','04:00','08:00','12:00']

      moon = get_body("moon", times)
      moonaltazs = moon.transform_to(frame_sun)

      target_altazs_new = target.transform_to(frame_sun)

      fig, ax = plt.subplots()
      ax.plot(delta_midnight_sun, sunaltazs.alt, color="#dc4424", label="Sun")
      ax.plot(delta_midnight_sun, moonaltazs.alt, color=[0.75] * 3, ls="--", label="Moon")
      im = ax.scatter(delta_midnight_sun, target_altazs_new.alt, c=target_altazs_new.az.value, label=star_name, lw=0, s=8, cmap=cm.batlow,)
      ax.fill_between(delta_midnight_sun, 0 * u.deg, 90 * u.deg, sunaltazs.alt < -0 * u.deg, color="0.5", zorder=0,)
      ax.fill_between(delta_midnight_sun, 0 * u.deg, 90 * u.deg, sunaltazs.alt < -18 * u.deg, color="k", zorder=0,)
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.2)
      fig.colorbar(im, cax=cax, orientation='vertical', label = "Azimuth [deg]")
      ax.legend(loc="upper left")
      ax.set_xlim(-12 * u.hour, 12 * u.hour)
      ax.set_xticks(np.linspace(-12, 12, 7) * u.hour)
      ax.set_xticklabels(labels)
      ax.set_ylim(0 * u.deg, 90 * u.deg)
      ax.set_xlabel("local time")
      ax.set_ylabel("Altitude [deg]")
      col1.pyplot(fig)

      with col1:
         with st.expander("click to read more"):
            st.write("The daily chart shows the altitude with time of the Sun, Moon and target - whose azimuth is also presented - on the selected day (UT offset: -3 hours). Near the horizon, the Earth's atmosphere is much thicker, which makes direct imaging challenging. This results in difficulties such as air turbulence and atmospheric refraction. Near zenith, the light from the object travels through much less air resulting in richer seeing. At 30 deg or higher, the observation is through 2 air masses or less of atmosphere. Below 30 deg, imaging is very difficult as the observation is through 3 air masses or more.")

    ##################################################################
      import vlti
      import observability

      col2.subheader('sky map')

      observe_time_sky = observe_time + np.linspace(-4, 4, 10)*u.hour

      target_styles = {'linestyle': '--', 'color': '#24448c'}

      fig = plot_sky(target_1, vlti_1, observe_time_sky, north_to_east_ccw=False, style_kwargs=target_styles).figure

      if option == "UT (UT1-UT2-UT3-UT4)":
         vlti.skyCoverage(['U1', 'U2', 'U3', 'U4'], max_OPD=100, fig=1, flexible=False,
                disp_vcmPressure=False, min_altitude=None,
                max_vcmPressure=None, verbose=False, STS=False,
                createXephemHorizon=False, plotIss=False,
                DLconstraints=None)

      if option == "AT-large (AO-G1-J2-K0)":
         vlti.skyCoverage(['A0', 'G1', 'J2', 'K0'], max_OPD=100, fig=1, flexible=False,
                disp_vcmPressure=False, min_altitude=None,
                max_vcmPressure=None, verbose=False, STS=False,
                createXephemHorizon=False, plotIss=False,
                DLconstraints=None)

      if option == "AT-medium (K0-G2-D0-J3)":
         vlti.skyCoverage(['K0', 'G2', 'D0', 'J3'], max_OPD=100, fig=1, flexible=False,
                disp_vcmPressure=False, min_altitude=None,
                max_vcmPressure=None, verbose=False, STS=False,
                createXephemHorizon=False, plotIss=False,
                DLconstraints=None)

      if option == "AT-small (A0-B2-D0-C1)":
         vlti.skyCoverage(['A0', 'B2', 'D0', 'C1'], max_OPD=100, fig=1, flexible=False,
                disp_vcmPressure=False, min_altitude=None,
                max_vcmPressure=None, verbose=False, STS=False,
                createXephemHorizon=False, plotIss=False,
                DLconstraints=None)
       
      plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
      col2.pyplot(fig)
      with col2:
         with st.expander("click to read more"):
            st.write("The sky map shows the movement of the target over an 8-hour period on the observation date. Sky coverage is limited by UT dome shadowing and delay line limits. To illustrate the effect of these contraints, this plot incorportes the VLTI sky coverage maps whose code was provided by Antoine Mérand.")


        
        #####################  CONFIGURATION ##############################

      col3.subheader('VLTI configuration')

      fig, ax = plt.subplots(figsize=(10,10))

      A0 = ax.plot(-14.642,-55.812, marker = '.', markersize=8, color='k')
      ax.text(-22,-55.812, 'A0')
      B2 = ax.plot(0.739,-75.899, marker = '.', markersize=5, color='k')
      ax.text(-7,-75.899, 'B2')
      D0 = ax.plot(15.628,-45.397, marker = '.', markersize=5, color='k')
      ax.text(8,-45.397, 'D0')
      C1 = ax.plot(5.691,-65.735, marker = '.', markersize=5, color='k')
      ax.text(-2,-65.735, 'C1')


      K0 = ax.plot(106.397,-14.165, marker = '.', markersize=5, color='k')
      ax.text(99,-14.165, 'K0')
      G2 = ax.plot(38.063,-12.289, marker = '.', markersize=5, color='k')
      ax.text(30,-12.289, 'G2')

      J3 = ax.plot(80.628,36.193, marker = '.', markersize=5, color='k')
      ax.text(76,36.193, 'J3')

      G1 = ax.plot(66.716,-95.501, marker = '.', markersize=5, color='k')
      ax.text(59,-95.501, 'G1')
      J2 = ax.plot(114.460,-62.151, marker = '.', markersize=5, color='k')
      ax.text(109,-62.151, 'J2')

      B5 = ax.plot(8.547,-98.594, marker = '.', markersize=5, color='k')
      ax.text(1,-98.594, 'B5')

      J6 = ax.plot(59.810,96.706, marker = '.', markersize=5, color='k')
      ax.text(55,96.706, 'J6')

      UT1 = ax.plot(-9.925,-20.335, marker = '.', markersize=36, color='k')
      ax.text(-25,-20.335, 'UT1')
      UT2 = ax.plot(14.887,30.502, marker = '.', markersize=36, color='k')
      ax.text(0,30.502, 'UT2')
      UT3 = ax.plot(44.915,66.183, marker = '.', markersize=36, color='k')
      ax.text(30,66.183, 'UT3')
      UT4 = ax.plot(103.306,43.999, marker = '.', markersize=36, color='k')
      ax.text(88,43.999, 'UT4')

      ax.set_xlim(-120, 120)
      ax.set_ylim(-120, 120)
      ax.set_xlabel("East coordinate (metres)", fontsize=20)
      ax.set_ylabel("North coordinate (metres)", fontsize=20)

      if option == "UT (UT1-UT2-UT3-UT4)":
         ax.plot([-9.925, 14.887], [-20.335,30.502], '#dc4424', label='UT1-UT2 (56.569 m)')  
         ax.plot([-9.925, 44.915], [-20.335,66.183], '#24448c', label='UT1-UT3 (102.434 m)')
         ax.plot([-9.925, 103.306], [-20.335,43.999], '#f8bc5c', label='UT1-UT4 (130.231 m)')
         ax.plot([14.887, 44.915], [30.502,66.183], '#74a466', label='UT2-UT3 (46.635 m)')
         ax.plot([14.887, 103.306], [30.502,43.999], '#cca4a0', label='UT2-UT4 (89.443 m)')  
         ax.plot([44.915, 103.306], [66.183,43.999], '#49443b', label='UT3-UT4 (62.463 m)')
         ax.set_title('UT (UT1-UT2-UT3-UT4)', fontsize=20)
         handles, labels = plt.gca().get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         ax.legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.5, fontsize='x-large')
         col3.pyplot(fig)
      
      if option == "AT-large (AO-G1-J2-K0)":
         ax.plot([-14.642, 66.716], [-55.812,-95.501], '#dc4424', label='A0-G1 (90.522 m)')  
         ax.plot([-14.642, 114.460], [-55.812,-62.151], '#24448c', label='A0-J2 (129.257 m)')
         ax.plot([-14.642, 106.397], [-55.812,-14.165], '#f8bc5c', label='A0-K0 (128.003 m)')
         ax.plot([66.716, 114.460], [-95.501,-62.151], '#74a466', label='G1-J2 (58.238 m)')
         ax.plot([66.716, 106.397], [-95.501,-14.165], '#cca4a0', label='G1-K0 (90.500 m)')  
         ax.plot([114.460, 106.397], [-62.151,-14.165], '#49443b', label='J2-K0 (48.659 m)')
         ax.set_title('AT-large (AO-G1-J2-K0)', fontsize=20)
         handles, labels = plt.gca().get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         ax.legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.5, fontsize='x-large')
         col3.pyplot(fig)

      if option == "AT-medium (K0-G2-D0-J3)":
         ax.plot([106.397, 38.063], [-14.165,-12.289], '#dc4424', label='K0-G2 (68.360 m)')  
         ax.plot([106.397, 15.628], [-14.165,-45.397], '#24448c', label='K0-D0 (95.992 m)')
         ax.plot([106.397, 80.628], [-14.165,36.193], '#f8bc5c', label='K0-J3 (56.569 m)')
         ax.plot([15.628, 38.063], [-45.397,-12.289], '#74a466', label='G2-D0 (39.993 m)')
         ax.plot([38.063, 80.628], [-12.289,36.193], '#cca4a0', label='G2-J3 (64.516 m)')  
         ax.plot([15.628, 80.628], [-45.397,36.193], '#49443b', label='D0-J3 (104.317 m)')
         handles, labels = plt.gca().get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         ax.legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.5, fontsize='x-large')
         ax.set_title('AT-medium (K0-G2-D0-J3)', fontsize=20)
         col3.pyplot(fig)

      if option == "AT-small (A0-B2-D0-C1)":
         ax.plot([-14.642, 0.739], [-55.812,-75.899], '#dc4424', label='A0-B2 (25.299 m)')
         ax.plot([-14.642, 15.628], [-55.812,-45.397], '#f8bc5c', label='A0-D0 (32.011 m)')
         ax.plot([-14.642, 5.691], [-55.812,-65.735], '#24448c', label='A0-C1 (22.625 m)')
         ax.plot([0.739, 15.628], [-75.899,-45.397], '#cca4a0', label='B2-D0 (33.941 m)')
         ax.plot([0.739, 5.691], [-75.899,-65.735], '#74a466', label='B2-C1 (11.306 m)')
         ax.plot([5.691, 15.628], [-65.735,-45.397], '#49443b', label='D0-C1 (22.635 m)')
         handles, labels = plt.gca().get_legend_handles_labels()
         by_label = dict(zip(labels, handles))
         ax.legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.5, fontsize='x-large')
         ax.set_title('AT-small (A0-B2-D0-C1)', fontsize=20)
         col3.pyplot(fig)

      with col3:
         with st.expander("click to read more"):
            st.write("The VLTI is comprised of an array of stations that are occupied by the 1.8-m auxiliary telescopes or the 8.2-m unit telescopes. This plot shows the chosen array configuration. The respective baseline lengths are shown in parentheses.")

        

        #####################  AIRMASS ###################################

      labels = ['22:00','00:00','02:00','04:00','06:00','08:00','10:00']

      vlti = EarthLocation.of_site('Paranal Observatory (ESO)')

      col4.subheader('airmass')
      delta_midnight_2 = np.linspace(-12, 12, 100) * u.hour
      frame_air_mass = AltAz(obstime=midnight + delta_midnight_2, location=vlti)
      target_altazs_air_mass = target.transform_to(frame_air_mass)

      target_airmasss = target_altazs_air_mass.secz

      fig, ax = plt.subplots()

      ax.plot(delta_midnight_2, target_airmasss, '#24448c', label=star_name)
      ax.set_xlim(-2, 10)
      ax.set_xticklabels(labels)
      ax.set_ylim(1, 4)
      ax.set_xlabel("local time")
      ax.set_ylabel("Airmass")
      ax.legend(loc='upper left')
      col4.pyplot(fig)

      with col4:
         with st.expander("click to read more"):
            st.write("This is a plot of the airmass as a function of time.")

      #col1.subheader('target image')

      #fig, ax = plt.subplots()

      #plot_finder_image(target_1)

      #col1.pyplot(fig)

      with col1:
         with st.expander("click to read more"):
            st.write("This is an image of the target.")


##################################################################

@st.cache_data
def get_plot_1(i):
   fig, ax = plt.subplots()
   im = ax.imshow(asim.maps[0, i, 3,:,:] - asim.maps[0, i, 4,:,:], extent=asim.map_extent, cmap=cm.romaO_r)
   fig.colorbar(im, label= 'Throughput $[m^2/sr]$')
   ax.set_xlabel("R.A. coordinate (mas)")
   ax.set_ylabel("Dec. coordinate (mas)")
   ax.plot([0.0, 0.0],[0.0, 0.0], "w*", ms=16)
   if option == "UT (UT1-UT2-UT3-UT4)":
      ax.set_title('UT1-UT2-UT3-UT4')
   if option == "AT-large (AO-G1-J2-K0)":
      ax.set_title('AO-G1-J2-K0')
   if option == "AT-medium (K0-G2-D0-J3)":
      ax.set_title('K0-G2-D0-J3')
   if option == "AT-small (A0-B2-D0-C1)":
      ax.set_title('A0-B2-D0-C1')
   return fig

@st.cache_data
def get_plot_2(i):
   fig, ax = plt.subplots()
   im = ax.imshow(asim.maps[0, i, 3,:,:] - asim.maps[0, i, 4,:,:], extent=asim.map_extent, cmap=cm.romaO_r)
   fig.colorbar(im, label= 'Throughput $[m^2/sr]$')
   ax.set_xlabel("R.A. coordinate (mas)")
   ax.set_ylabel("Dec. coordinate (mas)")
   ax.plot([0.0, 0.0],[0.0, 0.0], "w*", ms=16)
   if option == "UT (UT1-UT2-UT3-UT4)":
      ax.set_title('UT1-UT3-UT2-UT4')
   if option == "AT-large (AO-G1-J2-K0)":
      ax.set_title('AO-J2-G1-K0')
   if option == "AT-medium (K0-G2-D0-J3)":
      ax.set_title('K0-D0-G2-J3')
   if option == "AT-small (A0-B2-D0-C1)":
      ax.set_title('A0-D0-B2-C1')
   return fig

@st.cache_data
def get_plot_3(i):
   fig, ax = plt.subplots()
   im = ax.imshow(asim.maps[0, i, 3,:,:] - asim.maps[0, i, 4,:,:], extent=asim.map_extent, cmap=cm.romaO_r)
   fig.colorbar(im, label= 'Throughput $[m^2/sr]$')
   ax.set_xlabel("R.A. coordinate (mas)")
   ax.set_ylabel("Dec. coordinate (mas)")
   ax.plot([0.0, 0.0],[0.0, 0.0], "w*", ms=16)
   if option == "UT (UT1-UT2-UT3-UT4)":
      ax.set_title('UT1-UT4-UT3-UT2')
   if option == "AT-large (AO-G1-J2-K0)":
      ax.set_title('AO-K0-J2-G1')
   if option == "AT-medium (K0-G2-D0-J3)":
      ax.set_title('K0-J3-D0-G2')
   if option == "AT-small (A0-B2-D0-C1)":
      ax.set_title('A0-C1-D0-B2')
   return fig


if star_name:

   notebook_path = os.path.abspath("observability.py")
   config_file = os.path.join(os.path.dirname(notebook_path), "local_config/default_R400.ini")
   asim = makesim(config_file,target=star_name)
   
   with tab2:
      values = np.arange(0, 67, 1)
      labels = np.round(np.linspace(3.5, 4.0, 67), 4)
   
      index = st.select_slider('wavelength (in $\mu$m)', values, format_func=(lambda x:labels[x]), key='index')
    
      col1, col2, col3 = st.columns((1,1,1))

      swaps = np.array(["0,1,2,3", "0,2,1,3", "0,3,2,1"])

      if option == "UT (UT1-UT2-UT3-UT4)":
         asim.config.set("configuration", "config", value="VLTI")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = [get_plot_1(i) for i in range(67)]
         col1.pyplot(figs_1[index-1])
         plt.close(figs_1[index-1])

         asim.config.set("configuration", "config", value="VLTI")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)
         figs_2 = [get_plot_2(i) for i in range(67)]
         col2.pyplot(figs_2[index-1])
         plt.close(figs_2[index-1])

         asim.config.set("configuration", "config", value="VLTI")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1) 
         figs_3 = [get_plot_3(i) for i in range(67)]
         col3.pyplot(figs_3[index-1])
         plt.close(figs_3[index-1])

      if option == "AT-large (AO-G1-J2-K0)":

         asim.config.set("configuration", "config", value="VLTI_AT_large")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = [get_plot_1(i) for i in range(67)]
         col1.pyplot(figs_1[index-1])
         plt.close(figs_1[index-1])

         asim.config.set("configuration", "config", value="VLTI_AT_large")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)
         figs_2 = [get_plot_2(i) for i in range(67)]
         col2.pyplot(figs_2[index-1])
         plt.close(figs_2[index-1])

         asim.config.set("configuration", "config", value="VLTI_AT_large")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1) 
         figs_3 = [get_plot_3(i) for i in range(67)]
         col3.pyplot(figs_3[index-1])
         

      if option == "AT-medium (K0-G2-D0-J3)":

         asim.config.set("configuration", "config", value="VLTI_AT_medium")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = [get_plot_1(i) for i in range(67)]
         col1.pyplot(figs_1[index-1])
         plt.close(figs_1[index-1])

         asim.config.set("configuration", "config", value="VLTI_AT_medium")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)
         figs_2 = [get_plot_2(i) for i in range(67)]
         col2.pyplot(figs_2[index-1])
         plt.close(figs_2[index-1])

         asim.config.set("configuration", "config", value="VLTI_AT_medium")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1) 
         figs_3 = [get_plot_3(i) for i in range(67)]
         col3.pyplot(figs_3[index-1])
      
      if option == "AT-small (A0-B2-D0-C1)":

         asim.config.set("configuration", "config", value="VLTI_AT_small")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = [get_plot_1(i) for i in range(67)]
         col1.pyplot(figs_1[index-1])
         plt.close(figs_1[index-1])

         asim.config.set("configuration", "config", value="VLTI_AT_small")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1)
         figs_2 = [get_plot_2(i) for i in range(67)]
         col2.pyplot(figs_2[index-1])
         plt.close(figs_2[index-1])

         asim.config.set("configuration", "config", value="VLTI_AT_small")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         asim.build_all_maps_dask(mapcrop=0.1) 
         figs_3 = [get_plot_3(i) for i in range(67)]
         col3.pyplot(figs_3[index-1])
      
      
      with st.expander("click to read more"):
         st.write("NOTT/Double Bracewell throughput maps computed for three different input beam permutations. The order of the UTs/ATs is shown above each map. Taking the difference of nulled output values means that unlike raw photometric-like outputs, these maps can take negative values (shown by the colour scale on the right of the map where the throughput is expressed in units of $m^2/sr$). A white star in each map marks the location of the central star where the transmission, by design, is equal to zero. These maps are computed with [SCIFYsim](%s) - an end-to-end simulator designed for the NOTT instrument, which takes into account the various sources of noise and their correlation." % link)


@st.cache_data
def get_SNR_1():
   
   planet_signal_map = (a.planet_photons(planet_mag, dit=dit, T=atemp)[None,:, None,None,None] * asim.gain_map)
   difmap_flux = planet_signal_map[:,:,4,:,:] - planet_signal_map[:,:,3,:,:]
   snr_map = np.sqrt(((difmap_flux[:,:,:,:] / mysigs[None,:,None,None])**2).sum(axis=(0,1)))

   fig, ax = plt.subplots()
   im = ax.imshow(np.squeeze(dask.compute(snr_map), axis=0), cmap="cmc.lajolla", extent=asim.map_extent)
   fig.colorbar(im)#, label= 'Throughput $[m^2/sr]$')
   ax.set_xlabel("R.A. coordinate (mas)")
   ax.set_ylabel("Dec. coordinate (mas)")
   ax.scatter(planet_dec, planet_ra, color="w", marker="+")
   ax.plot([0.0, 0.0],[0.0, 0.0], "w*", ms=16)
   if option == "UT (UT1-UT2-UT3-UT4)":
      ax.set_title('UT1-UT2-UT3-UT4')
   if option == "AT-large (AO-G1-J2-K0)":
      ax.set_title('AO-G1-J2-K0')
   if option == "AT-medium (K0-G2-D0-J3)":
      ax.set_title('K0-G2-D0-J3')
   if option == "AT-small (A0-B2-D0-C1)":
      ax.set_title('A0-B2-D0-C1')
   return fig

@st.cache_data
def get_SNR_2():

   planet_signal_map = (a.planet_photons(planet_mag, dit=dit, T=atemp)[None,:, None,None,None] * asim.gain_map)
   difmap_flux = planet_signal_map[:,:,4,:,:] - planet_signal_map[:,:,3,:,:]
   snr_map = np.sqrt(((difmap_flux[:,:,:,:] / mysigs[None,:,None,None])**2).sum(axis=(0,1)))

   fig, ax = plt.subplots()
   im = ax.imshow(np.squeeze(dask.compute(snr_map), axis=0), cmap="cmc.lajolla", extent=asim.map_extent)
   fig.colorbar(im)#, label= 'Throughput $[m^2/sr]$')
   ax.set_xlabel("R.A. coordinate (mas)")
   ax.set_ylabel("Dec. coordinate (mas)")
   ax.scatter(planet_dec, planet_ra, color="w", marker="+")
   ax.plot([0.0, 0.0],[0.0, 0.0], "w*", ms=16)
   if option == "UT (UT1-UT2-UT3-UT4)":
      ax.set_title('UT1-UT3-UT2-UT4')
   if option == "AT-large (AO-G1-J2-K0)":
      ax.set_title('AO-J2-G1-K0')
   if option == "AT-medium (K0-G2-D0-J3)":
      ax.set_title('K0-D0-G2-J3')
   if option == "AT-small (A0-B2-D0-C1)":
      ax.set_title('A0-D0-B2-C1')
   return fig

@st.cache_data
def get_SNR_3():
   
   planet_signal_map = (a.planet_photons(planet_mag, dit=dit, T=atemp)[None,:, None,None,None] * asim.gain_map)
   difmap_flux = planet_signal_map[:,:,4,:,:] - planet_signal_map[:,:,3,:,:]
   snr_map = np.sqrt(((difmap_flux[:,:,:,:] / mysigs[None,:,None,None])**2).sum(axis=(0,1)))


   fig, ax = plt.subplots()
   im = ax.imshow(np.squeeze(dask.compute(snr_map), axis=0), cmap="cmc.lajolla", extent=asim.map_extent)
   fig.colorbar(im)#, label= 'Throughput $[m^2/sr]$')
   ax.set_xlabel("R.A. coordinate (mas)")
   ax.set_ylabel("Dec. coordinate (mas)")
   ax.scatter(planet_dec, planet_ra, color="w", marker="+")
   ax.plot([0.0, 0.0],[0.0, 0.0], "w*", ms=16)
   if option == "UT (UT1-UT2-UT3-UT4)":
      ax.set_title('UT1-UT4-UT3-UT2')
   if option == "AT-large (AO-G1-J2-K0)":
      ax.set_title('AO-K0-J2-G1')
   if option == "AT-medium (K0-G2-D0-J3)":
      ax.set_title('K0-J3-D0-G2')
   if option == "AT-small (A0-B2-D0-C1)":
      ax.set_title('A0-C1-D0-B2')
   return fig


if star_name:

   a = sf.analysis.BasicETC(asim)
   thepsig, thenoise, afig = a.show_signal_noise(13., dit=10., T=800., plot=True, show=False)
   atemp = 800.
   dit = 10.
   mysigs = thenoise*units.electron
   planet_mag = 20.
   
   with tab3:
      
      col1, col2, col3 = st.columns((1,1,1))
      
      
      if option == "UT (UT1-UT2-UT3-UT4)":
         loc_mas = (planet_ra, planet_dec) #(4.,4.)
         aloc = asim.get_loc_map(loc_mas)

         asim.config.set("configuration", "config", value="VLTI")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = get_SNR_1()
         col1.pyplot(figs_1)

         asim.config.set("configuration", "config", value="VLTI")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_2 = get_SNR_2()
         col2.pyplot(figs_2)

         asim.config.set("configuration", "config", value="VLTI")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_3 = get_SNR_3()
         col3.pyplot(figs_3)


      if option == "AT-large (AO-G1-J2-K0)":
         loc_mas = (planet_ra, planet_dec) #(4.,4.)
         aloc = asim.get_loc_map(loc_mas)

         asim.config.set("configuration", "config", value="VLTI_AT_large")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = get_SNR_1()
         col1.pyplot(figs_1)

         asim.config.set("configuration", "config", value="VLTI_AT_large")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_2 = get_SNR_2()
         col2.pyplot(figs_2)

         asim.config.set("configuration", "config", value="VLTI_AT_large")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_3 = get_SNR_3()
         col3.pyplot(figs_3)

      if option == "AT-medium (K0-G2-D0-J3)":
         loc_mas = (planet_ra, planet_dec) #(4.,4.)
         aloc = asim.get_loc_map(loc_mas)

         asim.config.set("configuration", "config", value="VLTI_AT_medium")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = get_SNR_1()
         col1.pyplot(figs_1)

         asim.config.set("configuration", "config", value="VLTI_AT_medium")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_2 = get_SNR_2()
         col2.pyplot(figs_2)

         asim.config.set("configuration", "config", value="VLTI_AT_medium")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_3 = get_SNR_3()
         col3.pyplot(figs_3)

      if option == "AT-small (A0-B2-D0-C1)":
         loc_mas = (planet_ra, planet_dec) #(4.,4.)
         aloc = asim.get_loc_map(loc_mas)

         asim.config.set("configuration", "config", value="VLTI_AT_small")
         asim.config.set("configuration", "order", value="0,1,2,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_1 = get_SNR_1()
         col1.pyplot(figs_1)

         asim.config.set("configuration", "config", value="VLTI_AT_small")
         asim.config.set("configuration", "order", value="0,2,1,3")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)

         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_2 = get_SNR_2()
         col2.pyplot(figs_2)

         asim.config.set("configuration", "config", value="VLTI_AT_small")
         asim.config.set("configuration", "order", value="0,3,2,1")
         asim.prepare_observatory(file=asim.config)
         asim.order = asim.obs.order
         asim.array = asim.obs.statlocs
         asim.point(asim.sequence[0], asim.target, disp_override=False, long_disp_override=False,)
         
         asim.build_all_maps_dask(mapcrop=0.1)   
         figs_3 = get_SNR_3()
         col3.pyplot(figs_3)
         
      with st.expander("click to read more"):
         st.write("NOTT/Double Bracewell SNR maps computed for three different input beam permutations. The order of the UTs/ATs is shown above each map. A white star in each map marks the location of the central star where the transmission, by design, is equal to zero. These maps are computed with [SCIFYsim](%s) - an end-to-end simulator designed for the NOTT instrument, which takes into account the various sources of noise and their correlation." % link)
       
