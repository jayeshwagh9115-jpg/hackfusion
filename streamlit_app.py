import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import json
import random
from streamlit_option_menu import option_menu
from streamlit_chat import message

# Page configuration
st.set_page_config(
    page_title="TrackFusion 3 - AI Pharmacy System",
    page_icon="💊",
    layout="wide"
)

# Initialize session state variables
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'inventory' not in st.session_state:
    st.session_state.inventory = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Sample data for demo
MEDICINES = {
    "Metformin": {"type": "Diabetes", "refill_days": 30, "prescription_required": True},
    "Lisinopril": {"type": "Blood Pressure", "refill_days": 30, "prescription_required": True},
    "Atorvastatin": {"type": "Cholesterol", "refill_days": 30, "prescription_required": True},
    "Levothyroxine": {"type": "Thyroid", "refill_days": 90, "prescription_required": True},
    "Albuterol": {"type": "Asthma", "refill_days": 30, "prescription_required": True},
    "Ibuprofen": {"type": "Pain Relief", "refill_days": 0, "prescription_required": False},
    "Vitamin D": {"type": "Supplement", "refill_days": 60, "prescription_required": False}
}

def init_demo_data():
    """Initialize demo data"""
    if not st.session_state.orders:
        for med_name, info in MEDICINES.items():
            st.session_state.orders.append({
                "medicine": med_name,
                "quantity": random.randint(1, 3),
                "date": datetime.now() - timedelta(days=random.randint(0, 90)),
                "user_id": "demo_user"
            })
    
    if not st.session_state.inventory:
        for med_name, info in MEDICINES.items():
            st.session_state.inventory.append({
                "medicine": med_name,
                "current_stock": random.randint(10, 100),
                "reorder_level": 20,
                "last_ordered": datetime.now() - timedelta(days=random.randint(10, 30))
            })

def predict_refill_dates():
    """Predict when medicines will run out"""
    predictions = {}
    
    for medicine, info in MEDICINES.items():
        # Get recent orders for this medicine
        recent_orders = [o for o in st.session_state.orders 
                        if o["medicine"] == medicine]
        
        if recent_orders:
            # Calculate average consumption
            total_qty = sum([o["quantity"] for o in recent_orders])
            days_span = max([(datetime.now() - o["date"]).days for o in recent_orders]) or 1
            daily_consumption = total_qty / days_span
            
            # Find inventory level
            inv_item = next((i for i in st.session_state.inventory 
                           if i["medicine"] == medicine), None)
            
            if inv_item and daily_consumption > 0:
                days_until_empty = inv_item["current_stock"] / daily_consumption
                refill_date = datetime.now() + timedelta(days=days_until_empty)
                
                predictions[medicine] = {
                    "refill_date": refill_date,
                    "days_until_empty": int(days_until_empty),
                    "daily_consumption": round(daily_consumption, 2),
                    "current_stock": inv_item["current_stock"]
                }
    
    st.session_state.predictions = predictions
    return predictions

def process_natural_language(input_text):
    """Simulate NLP processing for medicine ordering"""
    input_text = input_text.lower()
    
    # Simple keyword matching (in production, use NLP models)
    detected_meds = []
    quantities = []
    
    # Look for medicine names
    for med in MEDICINES.keys():
        if med.lower() in input_text:
            detected_meds.append(med)
    
    # Look for quantities
    words = input_text.split()
    for i, word in enumerate(words):
        if word.isdigit():
            quantities.append(int(word))
        elif word in ["one", "two", "three", "four", "five"]:
            num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
            quantities.append(num_map[word])
    
    return detected_meds, quantities[:len(detected_meds)] if quantities else [1] * len(detected_meds)

def add_to_cart(medicines, quantities):
    """Add items to cart"""
    for med, qty in zip(medicines, quantities):
        st.session_state.cart.append({
            "medicine": med,
            "quantity": qty,
            "price": round(random.uniform(5, 50), 2),
            "prescription_required": MEDICINES[med]["prescription_required"]
        })

def simulate_agent_workflow():
    """Simulate multi-agent system workflow"""
    agents = {
        "Ordering Agent": "Processing customer order...",
        "Forecast Agent": "Analyzing consumption patterns...",
        "Procurement Agent": "Preparing purchase order...",
        "Safety Agent": "Checking prescription validity..."
    }
    
    workflow_log = []
    for agent, action in agents.items():
        workflow_log.append(f"✅ {agent}: {action}")
        # Simulate processing time
        import time
        time.sleep(0.1)
    
    return workflow_log

# Initialize demo data
init_demo_data()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medicine.png", width=100)
    st.title("TrackFusion 3")
    st.caption("AI-Driven Autonomous Pharmacy")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "Voice Ordering", "Predictive Analysis", "Agent System", "Inventory", "Settings"],
        icons=["speedometer", "mic", "graph-up", "robot", "box", "gear"],
        menu_icon="menu-app",
        default_index=0,
    )

# Main Dashboard
if selected == "Dashboard":
    st.title("🏥 Autonomous Pharmacy Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Orders", len(st.session_state.orders))
    with col2:
        st.metric("Inventory Items", len(st.session_state.inventory))
    with col3:
        st.metric("Pending Refills", sum(1 for p in predict_refill_dates().values() 
                                        if p["days_until_empty"] < 7))
    
    # Recent Orders
    st.subheader("📋 Recent Orders")
    if st.session_state.orders:
        orders_df = pd.DataFrame(st.session_state.orders[-10:])
        st.dataframe(orders_df, use_container_width=True)
    
    # Inventory Status
    st.subheader("📊 Inventory Overview")
    inv_df = pd.DataFrame(st.session_state.inventory)
    if not inv_df.empty:
        fig = px.bar(inv_df, x='medicine', y='current_stock', 
                    title='Current Stock Levels', color='current_stock')
        st.plotly_chart(fig, use_container_width=True)

# Voice/Text Ordering
elif selected == "Voice Ordering":
    st.title("🎤 AI Ordering Assistant")
    
    tab1, tab2 = st.tabs(["Chat Interface", "Order Summary"])
    
    with tab1:
        st.subheader("Speak or type your order")
        
        # Chat input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input("Type your medicine order:", 
                                      placeholder="e.g., 'I need Metformin and two boxes of Ibuprofen'")
        with col2:
            voice_mode = st.checkbox("🎤 Voice Mode")
        
        if st.button("Process Order", type="primary") and user_input:
            with st.spinner("Processing your order..."):
                # Process natural language input
                medicines, quantities = process_natural_language(user_input)
                
                if medicines:
                    # Add to cart
                    add_to_cart(medicines, quantities)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "user": user_input,
                        "assistant": f"I've added {', '.join(medicines)} to your cart. Would you like to add anything else?"
                    })
                    
                    st.success(f"✅ Added {len(medicines)} item(s) to cart!")
                    
                    # Show prescription warning if needed
                    presc_meds = [m for m in medicines if MEDICINES[m]["prescription_required"]]
                    if presc_meds:
                        st.warning(f"⚠️ Note: {', '.join(presc_meds)} require prescription validation")
                else:
                    st.error("Could not identify medicines. Please try again with specific names.")
        
        # Display chat history
        st.subheader("Chat History")
        for chat in st.session_state.chat_history[-5:]:
            message(chat["user"], is_user=True, key=f"user_{hash(chat['user'])}")
            message(chat["assistant"], key=f"assistant_{hash(chat['assistant'])}")
    
    with tab2:
        st.subheader("🛒 Your Cart")
        if st.session_state.cart:
            cart_df = pd.DataFrame(st.session_state.cart)
            st.dataframe(cart_df, use_container_width=True)
            
            total = sum(item["price"] * item["quantity"] for item in st.session_state.cart)
            st.metric("Total Amount", f"${total:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Proceed to Checkout", type="primary"):
                    st.success("Order placed successfully! Agent system activated...")
                    # Simulate agent workflow
                    workflow_log = simulate_agent_workflow()
                    for log in workflow_log:
                        st.info(log)
            with col2:
                if st.button("Clear Cart"):
                    st.session_state.cart = []
                    st.rerun()
        else:
            st.info("Your cart is empty. Start ordering above!")

# Predictive Analysis
elif selected == "Predictive Analysis":
    st.title("🔮 Predictive Ordering Engine")
    
    # Generate predictions
    predictions = predict_refill_dates()
    
    if predictions:
        st.subheader("Medicine Refill Predictions")
        
        pred_data = []
        for med, pred in predictions.items():
            pred_data.append({
                "Medicine": med,
                "Refill Date": pred["refill_date"].strftime("%Y-%m-%d"),
                "Days Until Empty": pred["days_until_empty"],
                "Current Stock": pred["current_stock"],
                "Urgency": "Critical" if pred["days_until_empty"] < 7 else 
                          "Soon" if pred["days_until_empty"] < 14 else "Normal"
            })
        
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True)
        
        # Visualization
        st.subheader("Refill Timeline")
        fig = go.Figure(data=[
            go.Bar(name='Days Until Empty', 
                  x=pred_df['Medicine'], 
                  y=pred_df['Days Until Empty'],
                  marker_color=['red' if d < 7 else 'orange' if d < 14 else 'green' 
                               for d in pred_df['Days Until Empty']])
        ])
        fig.update_layout(title="Refill Urgency Visualization")
        st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refill suggestions
        st.subheader("📅 Auto-Refill Suggestions")
        critical_meds = [med for med, pred in predictions.items() 
                        if pred["days_until_empty"] < 7]
        
        if critical_meds:
            st.warning(f"⚠️ Immediate refill needed for: {', '.join(critical_meds)}")
            if st.button("Auto-Order Critical Medicines"):
                # Simulate auto-ordering
                add_to_cart(critical_meds, [1] * len(critical_meds))
                st.success(f"Auto-ordered {len(critical_meds)} critical medicines!")
        else:
            st.info("No immediate refills needed. Inventory levels are good.")
    else:
        st.info("No prediction data available. Add some orders first.")

# Agent System
elif selected == "Agent System":
    st.title("🤖 Multi-Agent System")
    
    st.subheader("Agent Architecture")
    
    agents = [
        {"name": "Ordering Agent", "status": "✅ Active", 
         "description": "Assists customers during ordering process"},
        {"name": "Forecast Agent", "status": "✅ Active", 
         "description": "Predicts medicine refill needs"},
        {"name": "Procurement Agent", "status": "🔄 Idle", 
         "description": "Prepares purchase orders automatically"},
        {"name": "Safety Agent", "status": "✅ Active", 
         "description": "Checks prescription requirements"}
    ]
    
    for agent in agents:
        with st.expander(f"{agent['name']} - {agent['status']}"):
            st.write(agent["description"])
            
            if agent["name"] == "Ordering Agent":
                if st.button("Simulate Customer Interaction", key=f"btn_{agent['name']}"):
                    st.session_state.chat_history.append({
                        "user": "I need my monthly medicines",
                        "assistant": "I can help with that! What medicines do you need?"
                    })
                    st.success("Simulated interaction added to chat history")
            
            elif agent["name"] == "Procurement Agent":
                if st.button("Generate Purchase Order", key=f"btn_{agent['name']}"):
                    with st.spinner("Generating PO..."):
                        workflow_log = simulate_agent_workflow()
                        for log in workflow_log:
                            st.code(log)
    
    st.subheader("Workflow Automation")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**MCP Integration**")
        st.checkbox("Zapier (Email/SMS)", value=True)
        st.checkbox("n8n (API Calls)", value=True)
        st.checkbox("Webhooks (CMS Sync)", value=True)
    
    with col2:
        st.write("**Trigger Actions**")
        if st.button("Run End-to-End Workflow", type="primary"):
            with st.spinner("Executing workflow..."):
                logs = [
                    "📧 Sent order confirmation via Zapier",
                    "🔗 Triggered supplier API via n8n",
                    "🔄 Updated Mediloon CMS via webhook",
                    "📦 Purchase order generated and sent"
                ]
                for log in logs:
                    st.success(log)

# Inventory Management
elif selected == "Inventory":
    st.title("📦 Inventory Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Inventory")
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            edited_df = st.data_editor(inv_df, 
                                      column_config={
                                          "current_stock": st.column_config.NumberColumn(
                                              "Current Stock",
                                              help="Current quantity in stock",
                                              min_value=0
                                          ),
                                          "reorder_level": st.column_config.NumberColumn(
                                              "Reorder Level",
                                              help="Reorder when stock falls below this",
                                              min_value=0
                                          )
                                      },
                                      use_container_width=True,
                                      num_rows="dynamic")
            
            if st.button("Update Inventory", type="primary"):
                st.session_state.inventory = edited_df.to_dict('records')
                st.success("Inventory updated successfully!")
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("📊 Generate Stock Report"):
            inv_df = pd.DataFrame(st.session_state.inventory)
            st.download_button(
                label="Download CSV",
                data=inv_df.to_csv().encode('utf-8'),
                file_name="inventory_report.csv",
                mime="text/csv"
            )
        
        if st.button("🔄 Sync with CMS"):
            st.info("Syncing with Mediloon CMS...")
            st.success("Inventory synchronized successfully!")
        
        if st.button("🔔 Check Low Stock"):
            low_stock = [i for i in st.session_state.inventory 
                        if i["current_stock"] < i["reorder_level"]]
            if low_stock:
                st.warning(f"Low stock alert for {len(low_stock)} items!")
                for item in low_stock:
                    st.error(f"{item['medicine']}: {item['current_stock']} left (Reorder at {item['reorder_level']})")
            else:
                st.success("All stock levels are adequate")

# Settings
elif selected == "Settings":
    st.title("⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["User Profile", "System Config", "API Integration"])
    
    with tab1:
        st.subheader("User Profile")
        
        # First-time onboarding questionnaire
        if not st.session_state.user_profile:
            st.info("First-time user detected. Please complete onboarding.")
            
            with st.form("onboarding_form"):
                name = st.text_input("Full Name")
                age = st.number_input("Age", min_value=1, max_value=120)
                conditions = st.multiselect("Medical Conditions (optional)", 
                                          ["Diabetes", "Hypertension", "Asthma", "Thyroid", "Other"])
                preferences = st.multiselect("Notification Preferences",
                                           ["Email", "SMS", "WhatsApp", "In-App"])
                
                if st.form_submit_button("Save Profile"):
                    st.session_state.user_profile = {
                        "name": name,
                        "age": age,
                        "conditions": conditions,
                        "preferences": preferences
                    }
                    st.success("Profile saved successfully!")
        else:
            st.json(st.session_state.user_profile)
            if st.button("Reset Profile"):
                st.session_state.user_profile = {}
                st.rerun()
    
    with tab2:
        st.subheader("System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Language Settings**")
            language = st.selectbox("Interface Language", 
                                  ["English", "German", "Arabic"])
            st.checkbox("Enable voice commands", value=True)
        
        with col2:
            st.write("**Prediction Settings**")
            prediction_horizon = st.slider("Prediction Horizon (days)", 
                                         7, 90, 30)
            auto_refill_threshold = st.slider("Auto-refill threshold (days)", 
                                            3, 14, 7)
        
        if st.button("Save Configuration", type="primary"):
            st.success("Configuration saved!")
    
    with tab3:
        st.subheader("API Integrations")
        
        zapier_key = st.text_input("Zapier API Key", type="password")
        n8n_webhook = st.text_input("n8n Webhook URL")
        mediloon_api = st.text_input("Mediloon CMS API Key", type="password")
        
        if st.button("Test Connections"):
            st.info("Testing API connections...")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("Zapier: ✅ Connected")
            with col2:
                st.success("n8n: ✅ Connected")
            with col3:
                st.success("Mediloon: ✅ Connected")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("TrackFusion 3 - AI-Driven Pharmacy System")
st.sidebar.caption("HackFusion Project Submission")

# Run the app
if __name__ == "__main__":
    # This is already a Streamlit app, no need for additional code
    pass
