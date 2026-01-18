#!/usr/bin/env python3

"""
Simple Streamlit GUI - streamlit version of temp_GUI (initial step).

Features (v1):
- Read/write job_orders.json
- Select an existing job and publish to /User_Input topic
- Add and publish new job JSONs
- Display recent messages from ROS2 topics (last X messages)

NOTE: This app works with ROS2 (rclpy) when available. If rclpy is not installed, the GUI remains functional for job editing but subscriptions/publishing are disabled.
"""

import os
import json
import threading
import time
from collections import deque

import streamlit as st

# Try to import ROS2; if not available, show message but keep GUI functional for job editing.
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False

# Paths
ROOT_DIR = os.path.dirname(__file__)
JOB_ORDERS_PATH = os.path.join(ROOT_DIR, "src", "system_nodes", "system_nodes", "job_orders.json")

# Example JSON template shown to user
EXAMPLE_JSON = {
    "job_name": "New Job Name",
    "job_ID": "3-4 letter code",
    "part_type": "metal_sheet",
    "material": "type of material (e.g., st37)",
    "part_thickness_mm": 1,
    "part_width_mm": 100,
    "part_weight_kg": 1,
    "process_order": ["bending"],
    "surface_quality_mm": 0,
    "tolerance_mm": 0
}

# Topics of interest for display
TOPIC_LIST = [
    'Sensors/hydraulic_press',
    'Sensors/process_pump',
    'Completed/hydraulic_press',
    'Job_Orders',
    'Control_CMD/process_pump',
    'Control_CMD/hydraulic_press',
    'Maintenance_Queue',
    # predictions and maintenance_feedback may also appear
    'Predictions/process_pump',
    'Predictions/hydraulic_press',
    'Production_Log',
    # Node status topic (used for health / readiness)
    '/Node_Status',
]

# Helper functions to read/write job orders

def read_job_orders():
    if not os.path.exists(JOB_ORDERS_PATH):
        return []
    try:
        with open(JOB_ORDERS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def write_job_orders(data):
    os.makedirs(os.path.dirname(JOB_ORDERS_PATH), exist_ok=True)
    with open(JOB_ORDERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


# ROS2 Node wrapper (lightweight, for subscriptions & publishing)
if ROS_AVAILABLE:
    class StreamlitTempGUINode(Node):
        def __init__(self, buffers):
            super().__init__('streamlit_gui_node')
            self.buffers = buffers
            self.lock = threading.Lock()

            # Publisher
            self.user_input_pub = self.create_publisher(String, 'User_Input', 10)

            # Subscriptions: for convenience we'll subscribe to the topics in TOPIC_LIST when available
            for topic in TOPIC_LIST:
                # All as std_msgs/String in this prototype
                try:
                    self.create_subscription(String, topic, self._make_callback(topic), 10)
                except Exception:
                    # skip topics that are invalid
                    pass

            self.get_logger().info('Streamlit GUI Node initialized.')

        def _make_callback(self, topic):
            def cb(msg):
                with self.lock:
                    if topic not in self.buffers:
                        self.buffers[topic] = deque(maxlen=100)
                    self.buffers[topic].appendleft((time.time(), msg.data))
            return cb

        def publish_user_input(self, job_order: dict):
            msg = String()
            msg.data = json.dumps(job_order)
            self.user_input_pub.publish(msg)
            # also log to a buffer for UI
            with self.lock:
                if 'User_Input' not in self.buffers:
                    self.buffers['User_Input'] = deque(maxlen=100)
                self.buffers['User_Input'].appendleft((time.time(), msg.data))


# Initialize session state
if 'buffers' not in st.session_state:
    st.session_state.buffers = {}

if 'last_published' not in st.session_state:
    st.session_state.last_published = None

if 'ros_running' not in st.session_state:
    st.session_state.ros_running = False


# Start ROS node in background if available
if ROS_AVAILABLE and not st.session_state.ros_running:
    try:
        rclpy.init()
        node = StreamlitTempGUINode(st.session_state.buffers)

        # Thread to spin node safely
        stop_flag = threading.Event()

        def spin_thread():
            try:
                while rclpy.ok() and not stop_flag.is_set():
                    rclpy.spin_once(node, timeout_sec=0.1)
            except Exception:
                pass

        t = threading.Thread(target=spin_thread, daemon=True)
        t.start()

        st.session_state.ros_node = node
        st.session_state.ros_thread = t
        st.session_state.ros_stop = stop_flag
        st.session_state.ros_running = True
    except Exception as e:
        st.warning(f'ROS2 could not be started: {e}')
        st.session_state.ros_running = False


# UI
st.set_page_config(page_title='Capstone Project - Predictive Maintenance Dashboard', layout='wide')
st.title('Capstone Project - Predictive Maintenance Dashboard')


# Node readiness indicators using /Node_Status topic
nodes = [
    ("Machine_Hydraulic_Press_Node", ["hydraulic_press_sensor_node"]),
    ("Machine_Process_Pump_Node", ["process_pump_sensor_node"]),
    ("Job_Scheduler_Node", ["job_scheduler_node"]),
    ("Predictor_Node", ["predictor_node"]),
    ("Controller_Node", ["controller_node"])
]

# Parse /Node_Status buffer to get latest status per node name
node_status_buf = st.session_state.buffers.get('/Node_Status', deque()) or st.session_state.buffers.get('Node_Status', deque())
latest_status = {}  # node_name -> (ts, status_str)
try:
    for ts, payload in list(node_status_buf):
        try:
            d = json.loads(payload)
            node_name = d.get('node')
            status_str = d.get('status')
            if node_name:
                prev = latest_status.get(node_name)
                if not prev or ts > prev[0]:
                    latest_status[node_name] = (ts, status_str)
        except Exception:
            # payload may already be just a simple string, ignore
            continue
except Exception:
    latest_status = {}

# Helper to choose icon
def status_icon_for(node_keys, now_ts, threshold=60):
    # node_keys is list of node_name strings to check
    best = None
    for nk in node_keys:
        if nk in latest_status:
            ts, stval = latest_status[nk]
            if not best or ts > best[0]:
                best = (ts, stval)
    if not best:
        return '⚪️', 'UNKNOWN'  # grey
    age = now_ts - best[0]
    stval = (best[1] or '').upper()
    if age > threshold:
        return '⚪️', f'{stval} (stale)'
    if stval == 'READY':
        return '🟢', 'READY'
    if stval in ('ERROR','NOT_READY','SHUTDOWN','FAIL'):
        return '🔴', stval
    # default grey for unknown statuses
    return '⚪️', stval or 'UNKNOWN'

now_ts = time.time()
cols = st.columns([1,1,1,1,1])
for (display_name, node_keys), col in zip(nodes, cols):
    icon, label = status_icon_for(node_keys, now_ts)
    col.markdown(f"**{display_name}**  \nStatus: {icon}")


# Main layout: two machine charts and controls column
col_a, col_b, col_c = st.columns([3,3,2])

with col_a:
    st.subheader('HYDRAULIC PRESS MACHINE')
    st.write('Placeholder for RUL chart')
    hp_chart_placeholder = st.empty()
    st.markdown('---')
    st.subheader('Machine Status')
    # Minimal status info
    hp_buf = st.session_state.buffers.get('Sensors/hydraulic_press', deque())
    if hp_buf:
        ts, payload = hp_buf[0]
        st.write('Last sensor:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)))
        st.text(payload)
    else:
        st.info('No sensor data for hydraulic press.')

    # Control status (from Control_CMD/hydraulic_press)
    try:
        ctrl_buf = st.session_state.buffers.get('Control_CMD/hydraulic_press', deque())
        if ctrl_buf and len(ctrl_buf) > 0:
            c_ts, c_payload = ctrl_buf[0]
            try:
                c = json.loads(c_payload)
                cmd = (c.get('command') or c.get('cmd') or '').upper()
                recovery = c.get('recovery_time_min') or c.get('recovery_time')
            except Exception:
                cmd = str(c_payload)
                recovery = None

            if isinstance(cmd, str):
                if cmd == 'NORMAL_OPERATION':
                    label = 'READY'
                    color = 'green'
                elif cmd == 'SLOW_DOWN':
                    label = 'MAINTENANCE - SLOW_DOWN'
                    color = 'orange'
                elif cmd == 'SHUTDOWN':
                    label = 'MAINTENANCE - SHUTDOWN'
                    color = 'red'
                else:
                    label = cmd or 'UNKNOWN'
                    color = 'black'

                st.markdown(f"**Control Status:** <span style='color:{color}; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
                if recovery is not None:
                    st.caption(f"Recovery time (min): {recovery}")
        else:
            st.info('No control commands received.')
    except Exception:
        st.info('Control status unavailable.')

with col_b:
    st.subheader('PROCESS PUMP MACHINE')
    st.write('Placeholder for RUL chart')
    pump_chart_placeholder = st.empty()
    st.markdown('---')
    st.subheader('Machine Status')
    pump_buf = st.session_state.buffers.get('Sensors/process_pump', deque())
    if pump_buf:
        ts, payload = pump_buf[0]
        st.write('Last sensor:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)))
        st.text(payload)
    else:
        st.info('No sensor data for process pump.')

    # Control status (from Control_CMD/process_pump)
    try:
        ctrl_buf = st.session_state.buffers.get('Control_CMD/process_pump', deque())
        if ctrl_buf and len(ctrl_buf) > 0:
            c_ts, c_payload = ctrl_buf[0]
            try:
                c = json.loads(c_payload)
                cmd = (c.get('command') or c.get('cmd') or '').upper()
                recovery = c.get('recovery_time_min') or c.get('recovery_time')
            except Exception:
                cmd = str(c_payload)
                recovery = None

            if isinstance(cmd, str):
                if cmd == 'NORMAL_OPERATION':
                    label = 'READY'
                    color = 'green'
                elif cmd == 'SLOW_DOWN':
                    label = 'MAINTENANCE - SLOW_DOWN'
                    color = 'orange'
                elif cmd == 'SHUTDOWN':
                    label = 'MAINTENANCE - SHUTDOWN'
                    color = 'red'
                else:
                    label = cmd or 'UNKNOWN'
                    color = 'black'

                st.markdown(f"**Control Status:** <span style='color:{color}; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
                if recovery is not None:
                    st.caption(f"Recovery time (min): {recovery}")
        else:
            st.info('No control commands received.')
    except Exception:
        st.info('Control status unavailable.')

with col_c:
    st.subheader('CONTROLS')
    st.write('Start Existing Job')
    jobs = read_job_orders()
    job_names = [j.get('job_name') for j in jobs] if jobs else []
    job_options = ['Select Job'] + job_names
    selected_job = st.selectbox('Choose job (existing)', job_options, index=0)
    # Create job via a structured form (expander)
    with st.expander('Create new job (form)'):
        with st.form('create_job_form'):
            jf_name = st.text_input('Job name', value='')
            jf_id = st.text_input('Job ID', value='')
            jf_part_type = st.text_input('Part type', value='metal_sheet')
            jf_material = st.text_input('Material', value='st37')
            jf_thickness = st.number_input('Part thickness (mm)', min_value=0.0, value=1.0, format="%.3f")
            jf_width = st.number_input('Part width (mm)', min_value=0.0, value=100.0, format="%.3f")
            jf_weight = st.number_input('Part weight (kg)', min_value=0.0, value=1.0, format="%.3f")
            jf_process = st.multiselect('Process order', ['bending','forming','drilling','grooving','pocketing','assembling','quality_control'], default=['bending'])
            jf_surface = st.number_input('Surface quality (mm)', min_value=0.0, value=0.0, format="%.3f")
            jf_tolerance = st.number_input('Tolerance (mm)', min_value=0.0, value=0.0, format="%.3f")
            jf_produce_amount = st.number_input('Production amount', min_value=1, value=100)
            jf_priority = st.selectbox('Priority', ('low','medium','high'))
            jf_mode = st.selectbox('Mode', ('FAST','REALTIME'))
            save_job = st.checkbox('Save job to job_orders.json', value=True)

            create_and_publish = st.form_submit_button('Create & Publish')

            if create_and_publish:
                # minimal validation
                missing = []
                if not jf_name.strip():
                    missing.append('job_name')
                if not jf_id.strip():
                    missing.append('job_ID')
                if not jf_material.strip():
                    missing.append('material')
                if not jf_process:
                    missing.append('process_order')

                if missing:
                    st.error(f'Missing fields: {missing}')
                else:
                    job_order = {
                        'job_name': jf_name.strip(),
                        'job_ID': jf_id.strip(),
                        'part_type': jf_part_type.strip(),
                        'material': jf_material.strip(),
                        'part_thickness_mm': float(jf_thickness),
                        'part_width_mm': float(jf_width),
                        'part_weight_kg': float(jf_weight),
                        'process_order': jf_process,
                        'surface_quality_mm': float(jf_surface),
                        'tolerance_mm': float(jf_tolerance),
                        'produce_amount': int(jf_produce_amount),
                        'priority': jf_priority,
                        'mode': jf_mode
                    }

                    if save_job:
                        jobs = read_job_orders()
                        jobs.append(job_order)
                        write_job_orders(jobs)
                        st.success('Job saved to job orders')

                    # Publish immediately
                    if ROS_AVAILABLE and st.session_state.ros_running:
                        try:
                            st.session_state.ros_node.publish_user_input(job_order)
                            st.session_state.last_published = job_order
                            st.success('Job created and published to /User_Input')
                        except Exception as e:
                            st.error(f'Publish error: {e}')
                    else:
                        st.info('ROS not available - job prepared locally')

    produce_amount_selected = st.number_input('Production amount (override)', min_value=1, value=100, step=1)
    job_priority = st.selectbox('Enter priority', ('low','medium','high'))
    sim_mode = st.radio('Select simulation mode', ('FAST','REALTIME'))

    if st.button('START JOB'):
        if selected_job and selected_job != '--':
            job = next((j for j in jobs if j.get('job_name') == selected_job), None)
            if job:
                job_to_publish = dict(job)
                # override or set the production amount chosen by user
                try:
                    job_to_publish['produce_amount'] = int(produce_amount_selected)
                except Exception:
                    job_to_publish['produce_amount'] = job_to_publish.get('produce_amount', 100)

                job_to_publish['priority'] = job_priority
                job_to_publish['mode'] = sim_mode

                if ROS_AVAILABLE and st.session_state.ros_running:
                    try:
                        st.session_state.ros_node.publish_user_input(job_to_publish)
                        st.success('Started job (published to /User_Input)')
                    except Exception as e:
                        st.error(f'Publish error: {e}')
                else:
                    st.info('ROS not available - job prepared locally')
            else:
                st.error('Selected job not found in job orders.')
        else:
            st.warning('No job selected. Choose an existing job or create a new one.')
    st.markdown('---')
    st.write('Manual Maintenance (publish)')
    machine_choice = st.selectbox('Select machine', ('hydraulic_press','process_pump'))
    recovery_time = st.number_input('Recovery time (s)', min_value=0, value=60)

    if st.button('Publish Maintenance'):
        msg = {"machine": machine_choice, "recovery_time": int(recovery_time)}
        if ROS_AVAILABLE and st.session_state.ros_running:
            try:
                # ensure maintenance publisher exists
                try:
                    st.session_state.ros_node.publish_maintenance(json.dumps(msg))
                except Exception:
                    # fallback: create a publisher and publish
                    pub = st.session_state.ros_node.create_publisher(__import__('std_msgs').msg.String, 'Maintenance_Queue', 10)
                    m = __import__('std_msgs').msg.String(); m.data = json.dumps(msg)
                    pub.publish(m)
                st.success('Maintenance message published')
            except Exception as e:
                st.error(f'Publish error: {e}')
        else:
            st.info('ROS not available - maintenance message prepared locally')

# Topic messages (full-width) with always-on auto-refresh
st.markdown('---')
st.subheader('Topic Messages (Live Monitor)')


def render_buffer(topic):
    buf = list(st.session_state.buffers.get(topic, deque()))
    if not buf:
        st.info('No messages.')
        return

    # Show the most recent N messages inline and put the rest in a scrollable expander
    N = 10
    latest = buf[:N]
    for ts, payload in latest:
        tstr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
        st.write(f'[{tstr}] {payload}')

    if len(buf) > N:
        older = buf[N:]
        # older contains older messages (newest-first), reverse to show oldest at top inside the expander
        older_lines = []
        for ts, payload in reversed(older):
            tstr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
            older_lines.append(f'[{tstr}] {payload}')
        safe_topic_key = topic.replace('/', '_').replace(' ', '_')
        with st.expander(f'Show older messages ({len(older)})'):
            st.text_area('Older messages', value='\n'.join(older_lines), height=200, key=f'older_{safe_topic_key}')

# Helper: build RUL time series (predicted vs ground truth) from buffers
def _extract_pred_and_prod_series(machine, max_points=200):
    """Return a pandas.DataFrame indexed by cycle with columns 'predicted_rul_min' and 'ground_truth_rul_min',
    or a dict fallback {'cycles', 'predicted_rul_min', 'ground_truth_rul_min'} when pandas is not available, or None if no data."""
    try:
        pred_buf = list(st.session_state.buffers.get(f'Predictions/{machine}', deque()))
        prod_buf = list(st.session_state.buffers.get('Production_Log', deque()))
        pred_map = {}
        prod_map = {}

        for ts, payload in pred_buf:
            try:
                d = json.loads(payload)
                if d.get('machine') != machine:
                    continue
                cycle = d.get('cycle')
                rul = d.get('rul')
                rul_min = None
                if isinstance(rul, dict):
                    rul_min = rul.get('rul_min') or rul.get('rul') or rul.get('rul_minutes')
                elif isinstance(rul, (int, float, str)):
                    try:
                        rul_min = float(rul)
                    except Exception:
                        rul_min = None
                if cycle is None or rul_min is None:
                    continue
                pred_map[int(cycle)] = float(rul_min)
            except Exception:
                continue

        for ts, payload in prod_buf:
            try:
                d = json.loads(payload)
                if d.get('machine') != machine:
                    continue
                cycle = d.get('cycle')
                rul_min = d.get('current_rul_min')
                if cycle is None or rul_min is None:
                    continue
                prod_map[int(cycle)] = int(rul_min)
            except Exception:
                continue

        cycles = sorted(set(list(pred_map.keys()) + list(prod_map.keys())))
        if not cycles:
            return None
        cycles = cycles[-max_points:]
        data = {'predicted_rul_min': [], 'ground_truth_rul_min': []}
        for c in cycles:
            data['predicted_rul_min'].append(pred_map.get(c, None))
            data['ground_truth_rul_min'].append(prod_map.get(c, None))

        try:
            import pandas as pd
            df = pd.DataFrame(data, index=cycles)
            df.index.name = 'cycle'
            return df
        except Exception:
            return {'cycles': cycles, **data}
    except Exception:
        return None

# Chart rendering helper
def _render_rul_chart(placeholder, data, title:str, predicted_color='orange', ground_truth_color='steelblue'):
    """Render RUL chart with Altair if available (allows setting predicted line color),
    otherwise fall back to st.line_chart."""
    try:
        # If pandas DataFrame, use Altair for nicer control
        import pandas as pd
        try:
            import altair as alt
            has_altair = True
        except Exception:
            has_altair = False

        if isinstance(data, dict):
            df = pd.DataFrame({'cycle': data['cycles'],
                               'predicted_rul_min': data['predicted_rul_min'],
                               'ground_truth_rul_min': data['ground_truth_rul_min']})
            df = df.set_index('cycle')
        else:
            df = data.copy()

        if has_altair:
            df_reset = df.reset_index().melt(id_vars=['cycle'], var_name='series', value_name='rul')
            base = alt.Chart(df_reset).encode(x='cycle:O')
            pred_line = base.transform_filter("datum.series == 'predicted_rul_min'").mark_line().encode(
                y='rul:Q',
                color=alt.value(predicted_color),
                tooltip=['cycle', 'series', 'rul']
            )
            gt_line = base.transform_filter("datum.series == 'ground_truth_rul_min'").mark_line().encode(
                y='rul:Q',
                color=alt.value(ground_truth_color),
                tooltip=['cycle', 'series', 'rul']
            )
            chart = (pred_line + gt_line).properties(title=title)
            placeholder.altair_chart(chart, width='stretch')
            return
        else:
            # fallback to simple st.line_chart
            if isinstance(data, dict):
                placeholder.line_chart({
                    'predicted_rul_min': data['predicted_rul_min'],
                    'ground_truth_rul_min': data['ground_truth_rul_min']
                })
            else:
                placeholder.line_chart(data)
            return
    except Exception:
        try:
            placeholder.info('Unable to render RUL chart')
        except Exception:
            pass

if hasattr(st, "fragment"):
    @st.fragment(run_every=1)
    def show_live_logs():
        col_refresh, col_status = st.columns([3, 1])
        with col_refresh:
            # Topic choice must be inside the fragment so it can be read on each refresh
            topic_choice = st.selectbox('Select topic to view', ['All'] + TOPIC_LIST + ['User_Input'], key="live_topic_selector")
        
        with col_status:
            st.caption(f"Last update: {time.strftime('%H:%M:%S')}")
            st.caption("Live (auto-refresh)")

        # Update RUL charts for both machines (predictions and ground truth)
        try:
            hp_data = _extract_pred_and_prod_series('hydraulic_press')
            if hp_data is None:
                hp_chart_placeholder.info('RUL data not available yet')
            else:
                _render_rul_chart(hp_chart_placeholder, hp_data, title='Hydraulic Press RUL', predicted_color='orange')
        except Exception:
            try:
                hp_chart_placeholder.info('RUL chart update failed')
            except Exception:
                pass

        try:
            pump_data = _extract_pred_and_prod_series('process_pump')
            if pump_data is None:
                pump_chart_placeholder.info('RUL data not available yet')
            else:
                _render_rul_chart(pump_chart_placeholder, pump_data, title='Process Pump RUL', predicted_color='orange')
        except Exception:
            try:
                pump_chart_placeholder.info('RUL chart update failed')
            except Exception:
                pass

        if topic_choice == 'All':
            # Show only topics that have data
            active_topics = [t for t in st.session_state.buffers.keys() if len(st.session_state.buffers[t]) > 0]
            if not active_topics:
                st.info("Waiting for data on topics...")
            for topic in active_topics:
                st.markdown(f"**{topic}**")
                render_buffer(topic)
                st.divider()
        else:
            st.subheader(topic_choice)
            render_buffer(topic_choice)

    # Call the function to render it on screen
    show_live_logs()

else:
    # If Streamlit is older than 1.37, show a warning
    st.error("Update Streamlit to version 1.37 or newer to enable live topic monitoring.")
    # Fallback: Manual button
    if st.button("Manual Refresh"):
        st.rerun()
    
    topic_choice = st.selectbox('Select topic to view', ['All'] + TOPIC_LIST + ['User_Input'])
    if topic_choice == 'All':
         for topic in list(st.session_state.buffers.keys()):
             st.subheader(topic)
             render_buffer(topic)
    else:
        render_buffer(topic_choice)

st.markdown('---')
st.markdown('###  ROS2 Node Graph')

# Try to use graphviz and components if available
try:
    import graphviz as gv
except Exception:
    gv = None

try:
    import streamlit.components.v1 as components
except Exception:
    components = None


def build_ros_graph_dot(node_obj=None):
    """Builds a DOT graph describing nodes and topics using ROS2 introspection.

    Strategy:
    - Use node_obj.get_topic_names_and_types() to list topics.
    - For each topic, call get_publishers_info_by_topic and get_subscriptions_info_by_topic to learn which nodes publish/subscribe.
    - Fallback to a simple static topology if introspection fails or is unavailable.
    Returns DOT source string or None.
    """
    try:
        if not ROS_AVAILABLE or node_obj is None:
            return None

        # Gather topics
        try:
            topics = node_obj.get_topic_names_and_types()
        except Exception:
            # method may not be available
            topics = []

        pub_map = {}
        sub_map = {}

        for t, _types in topics:
            try:
                pubs_info = node_obj.get_publishers_info_by_topic(t)
                subs_info = node_obj.get_subscriptions_info_by_topic(t)
            except Exception:
                # Some rclpy versions may name these differently or not expose them.
                pubs_info = []
                subs_info = []

            pubs = set()
            subs = set()
            for p in pubs_info:
                # Endpoint info objects usually have node_name attribute
                node_name = getattr(p, 'node_name', None) or getattr(p, 'node_name', None)
                if node_name:
                    pubs.add(node_name)
            for s in subs_info:
                node_name = getattr(s, 'node_name', None) or getattr(s, 'node_name', None)
                if node_name:
                    subs.add(node_name)

            pub_map[t] = pubs
            sub_map[t] = subs

        # If introspection returned nothing, fall back to static mapping
        if not pub_map and not sub_map:
            # fallback to previous static mapping
            pub_map = {
                'Sensors/hydraulic_press': {'hydraulic_press_sensor_node'},
                'Sensors/process_pump': {'process_pump_sensor_node'},
                'Completed/hydraulic_press': {'hydraulic_press_sensor_node'},
                'Job_Orders': {'job_scheduler_node'},
                'User_Input': {'streamlit_gui_node','job_scheduler_node'},
                'Predictions/process_pump': {'predictor_node'},
                'Control_CMD/hydraulic_press': {'controller_node'},
                'Control_CMD/process_pump': {'controller_node'},
                'Maintenance_Queue': {'controller_node'}
            }
            sub_map = {
                'Job_Orders': {'Machine_Hydraulic_Press_Node','Machine_Process_Pump_Node'},
                'Control_CMD/hydraulic_press': {'Machine_Hydraulic_Press_Node'},
                'Control_CMD/process_pump': {'Machine_Process_Pump_Node'},
                'User_Input': {'Job_Scheduler_Node'},
                'Completed/hydraulic_press': {'Job_Scheduler_Node'},
                'Predictions/process_pump': {'Controller_Node'},
                'Maintenance_Queue': {'Job_Scheduler_Node'}
            }

        # Build DOT
        dot_lines = ['digraph G {', 'rankdir=LR;', 'node [shape=box, style=filled, fillcolor="#f8f9fa"];']

        # Create node list from pub/sub maps
        node_names = set()
        for t, pubs in pub_map.items():
            node_names.update(pubs)
        for t, subs in sub_map.items():
            node_names.update(subs)

        # Add nodes
        for n in sorted(node_names):
            # style sensor nodes and controller differently if desired later
            dot_lines.append(f'"{n}" [shape=ellipse, fillcolor="#e6f7e6"];')

        # Add topic nodes and edges pub->topic and topic->sub
        for topic, pubs in pub_map.items():
            topic_node = f"topic_{topic}".replace('/', '_')
            label = topic
            dot_lines.append(f'"{topic_node}" [shape=box, fillcolor="#fff3cd", label="{label}"];')
            for p in pubs:
                dot_lines.append(f'"{p}" -> "{topic_node}";')
            # subscribe edges
            for s in sub_map.get(topic, set()):
                dot_lines.append(f'"{topic_node}" -> "{s}";')

        dot_lines.append('}')
        return '\n'.join(dot_lines)
    except Exception:
        return None


try:
    node_obj = None
    # our node reference is stored in session as 'ros_node'
    node_obj = st.session_state.get('ros_node')

    if node_obj:
        dot = build_ros_graph_dot(node_obj)
        if dot:
            # Zoom control (50% - 200%)
            zoom_pct = st.slider('Graph zoom %', 20, 50, 100, key='graph_zoom')

            # Try rendering DOT -> SVG and embed with scrollable container for zooming
            if gv is not None and components is not None:
                try:
                    src = gv.Source(dot)
                    svg = src.pipe(format='svg')
                    if isinstance(svg, bytes):
                        svg = svg.decode('utf-8')

                    scale = zoom_pct / 100.0
                    html = f"""
<div style="overflow:auto; border:1px solid #ddd; width:100%; height:600px;">
  <div style="transform:scale({scale}); transform-origin:0 0; display:inline-block;">
    {svg}
  </div>
</div>
"""
                    components.html(html, height=600, scrolling=True)
                except Exception as e:
                    st.warning(f'Graph rendering failed ({e}), falling back to static view')
                    st.graphviz_chart(dot, width='stretch')
            else:
                st.graphviz_chart(dot, width='stretch')
        else:
            st.info('Graph data not available yet')
    else:
        st.info('ROS node not initialized')
except Exception as e:
    st.error(f'Failed to build graph: {e}')

# small spacer
st.write('\n')


# Footer: ROS status and shutdown button
st.sidebar.header('System')
if ROS_AVAILABLE:
    st.sidebar.write('ROS2: Available')
    st.sidebar.write(f'ROS running: {st.session_state.ros_running}')
    if st.sidebar.button('Shutdown ROS Node'):
        try:
            if st.session_state.ros_running:
                st.session_state.ros_stop.set()
                # allow thread to exit
                time.sleep(0.2)
                try:
                    st.session_state.ros_node.destroy_node()
                except Exception:
                    pass
                try:
                    rclpy.shutdown()
                except Exception:
                    pass
                st.session_state.ros_running = False
                st.sidebar.success('ROS node Stopped.')
                try:
                    try:
                        st.set_query_params(autorefresh_ts=str(int(time.time() * 1000)))
                    except Exception:
                        st.set_query_params(autorefresh_ts=str(int(time.time() * 1000)))
                except Exception:
                    try:
                        js_reload = "<script>window.location.reload();</script>"
                        if components is not None:
                            components.html(js_reload, height=0)
                        else:
                            st.markdown(js_reload, unsafe_allow_html=True)
                    except Exception:
                        pass
        except Exception as e:
            st.sidebar.error(f'Error: {e}')
else:
    st.sidebar.warning('ROS2 (rclpy) could not be imported; subscriptions and publishing disabled.')


st.markdown('---')
st.caption('Note: This is a simple initial version (MSc Capstone). Next steps include real-time updates, better error handling, and automatic topic discovery.')
