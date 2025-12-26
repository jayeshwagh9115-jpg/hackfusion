import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import time

# Try to import optional dependencies with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Some visualizations will be simplified.")

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except ImportError:
    OPTION_MENU_AVAILABLE = False
    st.warning("streamlit-option-menu not available. Using tabs instead.")

# Page configuration
st.set_page_config(
    page_title="TrackFusion 3 - AI-Driven Autonomous Pharmacy System",
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
if 'agents' not in st.session_state:
    st.session_state.agents = {
        "Ordering Agent": {"status": "Active", "last_active": datetime.now()},
        "Forecast Agent": {"status": "Active", "last_active": datetime.now()},
        "Procurement Agent": {"status": "Idle", "last_active": datetime.now()},
        "Safety Agent": {"status": "Active", "last_active": datetime.now()}
    }

# Sample data for demo
MEDICINES = {
    "Metformin": {"type": "Diabetes", "refill_days": 30, "prescription_required": True, "price": 25.50},
    "Lisinopril": {"type": "Blood Pressure", "refill_days": 30, "prescription_required": True, "price": 18.75},
    "Atorvastatin": {"type": "Cholesterol", "refill_days": 30, "prescription_required": True, "price": 32.00},
    "Levothyroxine": {"type": "Thyroid", "refill_days": 90, "prescription_required": True, "price": 15.25},
    "Albuterol": {"type": "Asthma", "refill_days": 30, "prescription_required": True, "price": 28.50},
    "Ibuprofen": {"type": "Pain Relief", "refill_days": 0, "prescription_required": False, "price": 8.99},
    "Vitamin D": {"type": "Supplement", "refill_days": 60, "prescription_required": False, "price": 12.50},
    "Omeprazole": {"type": "Acid Reflux", "refill_days": 30, "prescription_required": True, "price": 22.00},
    "Amlodipine": {"type": "Blood Pressure", "refill_days": 30, "prescription_required": True, "price": 19.75},
    "Sertraline": {"type": "Antidepressant", "refill_days": 30, "prescription_required": True, "price": 35.00}
}

def init_demo_data():
    """Initialize demo data"""
    if not st.session_state.orders:
        for med_name, info in MEDICINES.items():
            for _ in range(random.randint(1, 4)):
                st.session_state.orders.append({
                    "id": f"ORD{random.randint(1000, 9999)}",
                    "medicine": med_name,
                    "quantity": random.randint(1, 3),
                    "date": datetime.now() - timedelta(days=random.randint(0, 120)),
                    "user_id": "demo_user",
                    "status": random.choice(["Delivered", "Processing", "Shipped"]),
                    "prescription_verified": info["prescription_required"] and random.choice([True, False])
                })
    
    if not st.session_state.inventory:
        for med_name, info in MEDICINES.items():
            st.session_state.inventory.append({
                "medicine": med_name,
                "category": info["type"],
                "current_stock": random.randint(5, 150),
                "reorder_level": random.randint(10, 30),
                "last_ordered": datetime.now() - timedelta(days=random.randint(1, 60)),
                "supplier": random.choice(["PharmaCorp", "MediSupply", "HealthPlus", "Global Pharma"]),
                "lead_time_days": random.randint(2, 7)
            })

def predict_refill_dates():
    """Predict when medicines will run out using consumption patterns"""
    predictions = {}
    
    for medicine, info in MEDICINES.items():
        # Get recent orders for this medicine (last 90 days)
        recent_orders = [o for o in st.session_state.orders 
                        if o["medicine"] == medicine and 
                        (datetime.now() - o["date"]).days <= 90]
        
        if recent_orders:
            # Calculate consumption patterns
            total_qty = sum([o["quantity"] for o in recent_orders])
            if len(recent_orders) > 1:
                dates = sorted([o["date"] for o in recent_orders])
                days_span = (dates[-1] - dates[0]).days or 1
            else:
                days_span = 30  # Default assumption
            
            daily_consumption = total_qty / max(days_span, 1)
            
            # Find inventory level
            inv_item = next((i for i in st.session_state.inventory 
                           if i["medicine"] == medicine), None)
            
            if inv_item and daily_consumption > 0:
                days_until_empty = inv_item["current_stock"] / daily_consumption
                refill_date = datetime.now() + timedelta(days=days_until_empty)
                
                # Calculate confidence based on data points
                confidence = min(95, 70 + (len(recent_orders) * 5))
                
                predictions[medicine] = {
                    "refill_date": refill_date,
                    "days_until_empty": int(days_until_empty),
                    "daily_consumption": round(daily_consumption, 3),
                    "current_stock": inv_item["current_stock"],
                    "reorder_level": inv_item["reorder_level"],
                    "confidence": confidence,
                    "urgency": "CRITICAL" if days_until_empty < 3 else 
                              "HIGH" if days_until_empty < 7 else 
                              "MEDIUM" if days_until_empty < 14 else "LOW"
                }
    
    st.session_state.predictions = predictions
    return predictions

def process_natural_language(input_text):
    """Advanced NLP simulation for medicine ordering"""
    input_text = input_text.lower().strip()
    
    # Common phrases mapping
    phrase_mapping = {
        "my diabetes meds": ["Metformin"],
        "blood pressure pills": ["Lisinopril", "Amlodipine"],
        "cholesterol medicine": ["Atorvastatin"],
        "pain relievers": ["Ibuprofen"],
        "thyroid medication": ["Levothyroxine"],
        "asthma inhaler": ["Albuterol"],
        "antidepressants": ["Sertraline"],
        "acid reflux meds": ["Omeprazole"]
    }
    
    detected_meds = []
    quantities = []
    
    # Check for phrases first
    for phrase, meds in phrase_mapping.items():
        if phrase in input_text:
            detected_meds.extend(meds)
    
    # If no phrase match, look for individual medicine names
    if not detected_meds:
        for med in MEDICINES.keys():
            if med.lower() in input_text:
                detected_meds.append(med)
    
    # Extract quantities with more sophisticated parsing
    quantity_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "some": 2, "few": 3, "several": 3
    }
    
    words = input_text.split()
    for i, word in enumerate(words):
        # Check for numeric quantities
        if word.isdigit():
            quantities.append(int(word))
        # Check for word quantities
        elif word in quantity_words:
            quantities.append(quantity_words[word])
        # Check for "X boxes/packs/bottles of"
        elif word in ["boxes", "box", "packs", "pack", "bottles", "bottle"]:
            if i > 0 and words[i-1].isdigit():
                quantities.append(int(words[i-1]))
    
    # Default quantity if none specified
    if not quantities and detected_meds:
        quantities = [1] * len(detected_meds)
    # If fewer quantities than medicines, pad with 1s
    elif len(quantities) < len(detected_meds):
        quantities.extend([1] * (len(detected_meds) - len(quantities)))
    
    return list(set(detected_meds)), quantities[:len(detected_meds)]

def add_to_cart(medicines, quantities):
    """Add items to cart with validation"""
    for med, qty in zip(medicines, quantities):
        # Check if already in cart
        existing_idx = next((i for i, item in enumerate(st.session_state.cart) 
                           if item["medicine"] == med), None)
        
        if existing_idx is not None:
            # Update quantity if already in cart
            st.session_state.cart[existing_idx]["quantity"] += qty
        else:
            # Add new item to cart
            st.session_state.cart.append({
                "medicine": med,
                "quantity": qty,
                "price": MEDICINES[med]["price"],
                "prescription_required": MEDICINES[med]["prescription_required"],
                "category": MEDICINES[med]["type"],
                "added_at": datetime.now()
            })

def simulate_agent_workflow(order_type="standard"):
    """Simulate multi-agent system workflow"""
    agents_log = []
    
    workflow_steps = {
        "Ordering Agent": [
            "Received customer request",
            "Parsed medicine requirements",
            "Validated order details",
            "Forwarded to inventory check"
        ],
        "Safety Agent": [
            "Checking prescription requirements",
            "Validating drug interactions",
            "Verifying dosage information",
            "Safety check completed"
        ],
        "Forecast Agent": [
            "Analyzing consumption patterns",
            "Predicting future demand",
            "Calculating reorder points",
            "Inventory forecast updated"
        ],
        "Procurement Agent": [
            "Generating purchase order",
            "Contacting suppliers",
            "Negotiating prices",
            "Order confirmed with supplier"
        ]
    }
    
    for agent, steps in workflow_steps.items():
        agents_log.append(f"🤖 **{agent}**")
        for step in steps:
            agents_log.append(f"  ↳ {step}")
            time.sleep(0.05)  # Simulate processing time
        agents_log.append("")
    
    # Add MCP integration simulation
    agents_log.append("🔗 **MCP Integration**")
    agents_log.append("  ↳ Connected to Zapier for notifications")
    agents_log.append("  ↳ Triggered n8n workflow for supplier API")
    agents_log.append("  ↳ Updated Mediloon CMS via webhook")
    agents_log.append("  ↳ Synchronized with pharmacy inventory system")
    
    return agents_log

def generate_order_summary():
    """Generate detailed order summary"""
    if not st.session_state.cart:
        return None
    
    summary = {
        "total_items": len(st.session_state.cart),
        "total_quantity": sum(item["quantity"] for item in st.session_state.cart),
        "subtotal": sum(item["price"] * item["quantity"] for item in st.session_state.cart),
        "prescription_items": sum(1 for item in st.session_state.cart if item["prescription_required"]),
        "regular_items": sum(1 for item in st.session_state.cart if not item["prescription_required"]),
        "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d, %Y"),
        "order_id": f"ORD{random.randint(10000, 99999)}"
    }
    
    summary["tax"] = round(summary["subtotal"] * 0.08, 2)
    summary["shipping"] = 5.99 if summary["subtotal"] < 50 else 0
    summary["total"] = summary["subtotal"] + summary["tax"] + summary["shipping"]
    
    return summary

# Initialize demo data
init_demo_data()

# Sidebar Navigation with custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
        margin-bottom: 1rem;
    }
    .agent-card {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bae6fd;
        margin-bottom: 1rem;
    }
    .urgency-critical { color: #dc2626; font-weight: bold; }
    .urgency-high { color: #ea580c; font-weight: bold; }
    .urgency-medium { color: #ca8a04; font-weight: bold; }
    .urgency-low { color: #16a34a; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="main-header">💊 TrackFusion 3</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Driven Autonomous Pharmacy</div>', unsafe_allow_html=True)
    
    if OPTION_MENU_AVAILABLE:
        selected = option_menu(
            menu_title="Navigation",
            options=["Dashboard", "AI Assistant", "Predictive Engine", "Agent System", "Inventory", "Workflow", "Settings"],
            icons=["speedometer", "robot", "graph-up-arrow", "people", "box-seam", "gear-wide-connected", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                "icon": {"color": "#2563eb", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#2563eb"},
            }
        )
    else:
        st.subheader("Navigation")
        selected = st.selectbox(
            "Select Page",
            ["Dashboard", "AI Assistant", "Predictive Engine", "Agent System", "Inventory", "Workflow", "Settings"]
        )

# Dashboard Page
if selected == "Dashboard":
    st.title("🏥 Autonomous Pharmacy Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", len(st.session_state.orders), delta=f"+{random.randint(1, 5)} today")
    with col2:
        st.metric("Active Inventory", len(st.session_state.inventory))
    with col3:
        predictions = predict_refill_dates()
        critical_count = sum(1 for p in predictions.values() if p["urgency"] in ["CRITICAL", "HIGH"])
        st.metric("Urgent Refills", critical_count, delta="Needs attention" if critical_count > 0 else "All good")
    with col4:
        st.metric("Agent System", "4 Active", "100% Uptime")
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    
    tab1, tab2, tab3 = st.tabs(["Recent Orders", "Inventory Status", "System Alerts"])
    
    with tab1:
        if st.session_state.orders:
            recent_orders = sorted(st.session_state.orders, key=lambda x: x["date"], reverse=True)[:10]
            orders_df = pd.DataFrame(recent_orders)
            
            # Format date for display
            orders_df["date"] = pd.to_datetime(orders_df["date"]).dt.strftime("%Y-%m-%d %H:%M")
            
            st.dataframe(
                orders_df[["id", "medicine", "quantity", "date", "status"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No orders found. Demo data will be generated automatically.")
            if st.button("Generate Demo Orders"):
                init_demo_data()
                st.rerun()
    
    with tab2:
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    inv_df.nlargest(10, "current_stock"),
                    x="medicine",
                    y="current_stock",
                    color="current_stock",
                    title="Top 10 Medicines by Stock Level",
                    labels={"current_stock": "Stock Level", "medicine": "Medicine"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(
                    inv_df[["medicine", "current_stock", "reorder_level", "supplier"]],
                    use_container_width=True
                )
                
                # Simple bar chart using Streamlit
                chart_data = inv_df.nlargest(10, "current_stock")[["medicine", "current_stock"]]
                st.bar_chart(chart_data.set_index("medicine"))
    
    with tab3:
        # Generate system alerts
        alerts = []
        
        # Check for low stock
        for item in st.session_state.inventory:
            if item["current_stock"] < item["reorder_level"]:
                alerts.append({
                    "type": "⚠️ Inventory Alert",
                    "message": f"{item['medicine']} stock low ({item['current_stock']} left)",
                    "priority": "High"
                })
        
        # Check for pending prescriptions
        presc_orders = [o for o in st.session_state.orders 
                       if o.get("prescription_required", False) and not o.get("prescription_verified", False)]
        if presc_orders:
            alerts.append({
                "type": "📋 Prescription Alert",
                "message": f"{len(presc_orders)} orders need prescription verification",
                "priority": "Medium"
            })
        
        # Check agent status
        inactive_agents = [name for name, data in st.session_state.agents.items() 
                          if data["status"] != "Active"]
        if inactive_agents:
            alerts.append({
                "type": "🤖 Agent Alert",
                "message": f"Agents inactive: {', '.join(inactive_agents)}",
                "priority": "High"
            })
        
        if alerts:
            for alert in alerts:
                with st.expander(f"{alert['type']} - {alert['priority']} Priority"):
                    st.write(alert["message"])
                    if alert["priority"] == "High":
                        st.button("Take Action", key=f"action_{alert['type']}")
        else:
            st.success("✅ All systems operational. No alerts at this time.")

# AI Assistant Page
elif selected == "AI Assistant":
    st.title("🎤 AI Ordering Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history[-10:]):
                if chat["role"] == "user":
                    st.chat_message("user").write(chat["content"])
                else:
                    st.chat_message("assistant").write(chat["content"])
        
        # Chat input
        user_input = st.chat_input("Type your medicine order or question...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process the input
            with st.spinner("🤖 AI Assistant is thinking..."):
                # Simulate AI processing
                time.sleep(0.5)
                
                # Process natural language
                medicines, quantities = process_natural_language(user_input)
                
                if medicines:
                    # Add to cart
                    add_to_cart(medicines, quantities)
                    
                    # Generate response
                    response = f"I've added {', '.join(medicines)} to your cart. "
                    
                    # Check for prescription requirements
                    presc_meds = [m for m in medicines if MEDICINES[m]["prescription_required"]]
                    if presc_meds:
                        response += f"\n\n⚠️ **Note:** {', '.join(presc_meds)} require prescription validation. "
                        response += "Please have your prescription ready for verification."
                    
                    # Suggest alternatives if available
                    if len(medicines) == 1:
                        med = medicines[0]
                        alternatives = [m for m in MEDICINES.keys() 
                                      if MEDICINES[m]["type"] == MEDICINES[med]["type"] and m != med]
                        if alternatives:
                            response += f"\n\n💡 **Alternative suggestion:** Consider {random.choice(alternatives)} as an alternative."
                    
                    response += "\n\nWould you like to add anything else to your order?"
                else:
                    # General conversation
                    responses = [
                        "I'm here to help you order medicines. You can say things like 'I need Metformin' or 'Order my blood pressure medication'.",
                        "I can help you refill prescriptions, suggest alternatives, and track your medicine consumption.",
                        "Please tell me what medicines you need, and I'll add them to your cart automatically."
                    ]
                    response = random.choice(responses)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    with col2:
        st.subheader("🛒 Your Cart")
        
        if st.session_state.cart:
            cart_df = pd.DataFrame(st.session_state.cart)
            
            # Display cart items
            for i, item in enumerate(st.session_state.cart):
                with st.expander(f"{item['medicine']} ({item['quantity']}x)"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Price:** ${item['price']:.2f}")
                        st.write(f"**Category:** {item['category']}")
                    with col_b:
                        st.write(f"**Subtotal:** ${item['price'] * item['quantity']:.2f}")
                        if item['prescription_required']:
                            st.write("**Prescription:** Required")
                        else:
                            st.write("**Prescription:** Not required")
                    
                    # Quantity adjustment
                    new_qty = st.number_input(
                        "Quantity",
                        min_value=0,
                        max_value=10,
                        value=item["quantity"],
                        key=f"qty_{i}"
                    )
                    
                    if new_qty != item["quantity"]:
                        if new_qty == 0:
                            st.session_state.cart.pop(i)
                            st.rerun()
                        else:
                            st.session_state.cart[i]["quantity"] = new_qty
                            st.rerun()
            
            # Order summary
            st.markdown("---")
            summary = generate_order_summary()
            
            if summary:
                st.write("**Order Summary:**")
                st.write(f"Items: {summary['total_items']}")
                st.write(f"Quantity: {summary['total_quantity']}")
                st.write(f"Subtotal: ${summary['subtotal']:.2f}")
                st.write(f"Tax: ${summary['tax']:.2f}")
                st.write(f"Shipping: ${summary['shipping']:.2f}")
                st.markdown(f"**Total: ${summary['total']:.2f}**")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    if st.button("🔄 Update Cart", use_container_width=True):
                        st.rerun()
                with col_y:
                    if st.button("✅ Checkout", type="primary", use_container_width=True):
                        # Simulate checkout process
                        with st.spinner("Processing your order..."):
                            workflow_log = simulate_agent_workflow()
                            
                            st.success("🎉 Order placed successfully!")
                            st.info(f"**Order ID:** {summary['order_id']}")
                            st.info(f"**Estimated Delivery:** {summary['estimated_delivery']}")
                            
                            st.subheader("🤖 Agent Workflow Activated:")
                            for log in workflow_log:
                                st.code(log)
                            
                            # Clear cart after successful order
                            st.session_state.cart = []
                            st.rerun()
            
            if st.button("🗑️ Clear Cart", type="secondary"):
                st.session_state.cart = []
                st.rerun()
        
        else:
            st.info("Your cart is empty. Start ordering using the chat interface!")

# Predictive Engine Page
elif selected == "Predictive Engine":
    st.title("🔮 Predictive Ordering Engine")
    
    # Generate predictions
    predictions = predict_refill_dates()
    
    if predictions:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            critical = sum(1 for p in predictions.values() if p["urgency"] == "CRITICAL")
            st.metric("Critical Refills", critical, delta="Needs immediate attention" if critical > 0 else "None")
        
        with col2:
            avg_confidence = np.mean([p["confidence"] for p in predictions.values()])
            st.metric("Prediction Confidence", f"{avg_confidence:.1f}%")
        
        with col3:
            avg_days = np.mean([p["days_until_empty"] for p in predictions.values()])
            st.metric("Avg Days Until Empty", f"{avg_days:.1f} days")
        
        # Detailed predictions table
        st.subheader("📊 Medicine Refill Predictions")
        
        pred_data = []
        for med, pred in predictions.items():
            pred_data.append({
                "Medicine": med,
                "Category": MEDICINES[med]["type"],
                "Current Stock": pred["current_stock"],
                "Reorder Level": pred["reorder_level"],
                "Daily Consumption": pred["daily_consumption"],
                "Days Until Empty": pred["days_until_empty"],
                "Refill Date": pred["refill_date"].strftime("%Y-%m-%d"),
                "Confidence": f"{pred['confidence']}%",
                "Urgency": pred["urgency"]
            })
        
        pred_df = pd.DataFrame(pred_data)
        
        # Sort by urgency
        urgency_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        pred_df["urgency_order"] = pred_df["Urgency"].map(urgency_order)
        pred_df = pred_df.sort_values("urgency_order").drop(columns="urgency_order")
        
        # Display with conditional formatting
        def color_urgency(val):
            if val == "CRITICAL":
                return "background-color: #fecaca"
            elif val == "HIGH":
                return "background-color: #fed7aa"
            elif val == "MEDIUM":
                return "background-color: #fef08a"
            else:
                return "background-color: #bbf7d0"
        
        styled_df = pred_df.style.applymap(color_urgency, subset=["Urgency"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Visualization
        st.subheader("📈 Prediction Visualization")
        
        tab1, tab2 = st.tabs(["Stock Timeline", "Consumption Patterns"])
        
        with tab1:
            if PLOTLY_AVAILABLE:
                # Create timeline visualization
                fig = go.Figure()
                
                for med, pred in predictions.items():
                    days_left = pred["days_until_empty"]
                    fig.add_trace(go.Bar(
                        x=[med],
                        y=[days_left],
                        name=med,
                        marker_color='red' if pred["urgency"] == "CRITICAL" else
                                    'orange' if pred["urgency"] == "HIGH" else
                                    'yellow' if pred["urgency"] == "MEDIUM" else 'green',
                        text=[f"{days_left} days"],
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title="Days Until Stock Depletion",
                    xaxis_title="Medicine",
                    yaxis_title="Days Remaining",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback visualization
                chart_data = pred_df[["Medicine", "Days Until Empty"]].set_index("Medicine")
                st.bar_chart(chart_data)
        
        with tab2:
            # Consumption analysis
            consumption_data = []
            for med, pred in predictions.items():
                consumption_data.append({
                    "Medicine": med,
                    "Daily Consumption": pred["daily_consumption"],
                    "Monthly Consumption": pred["daily_consumption"] * 30,
                    "Stock Cover (days)": pred["days_until_empty"]
                })
            
            cons_df = pd.DataFrame(consumption_data)
            st.dataframe(cons_df, use_container_width=True)
        
        # Auto-refill recommendations
        st.subheader("🤖 Auto-Refill Recommendations")
        
        # Get medicines needing refill
        needs_refill = [med for med, pred in predictions.items() 
                       if pred["urgency"] in ["CRITICAL", "HIGH"]]
        
        if needs_refill:
            st.warning(f"⚠️ **Immediate Action Required:** {len(needs_refill)} medicines need urgent refill")
            
            for med in needs_refill:
                pred = predictions[med]
                with st.expander(f"🔴 {med} - {pred['urgency']} Priority"):
                    st.write(f"**Current Stock:** {pred['current_stock']}")
                    st.write(f"**Days Until Empty:** {pred['days_until_empty']}")
                    st.write(f"**Recommended Order Quantity:** {max(30, int(pred['daily_consumption'] * 30))}")
                    st.write(f"**Suggested Refill Date:** {pred['refill_date'].strftime('%Y-%m-%d')}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Auto-Order {med}", key=f"auto_{med}"):
                            add_to_cart([med], [max(2, int(pred['daily_consumption'] * 30))])
                            st.success(f"Added {med} to cart for auto-refill!")
                    with col_b:
                        if st.button(f"Schedule {med}", key=f"sched_{med}"):
                            st.info(f"{med} scheduled for refill on {pred['refill_date'].strftime('%Y-%m-%d')}")
            
            if st.button("🔄 Auto-Order All Critical Items", type="primary"):
                quantities = []
                for med in needs_refill:
                    pred = predictions[med]
                    quantities.append(max(2, int(pred['daily_consumption'] * 30)))
                
                add_to_cart(needs_refill, quantities)
                st.success(f"Added {len(needs_refill)} critical medicines to cart!")
        else:
            st.success("✅ No immediate refills needed. All inventory levels are satisfactory.")
        
        # Prediction settings
        with st.expander("⚙️ Prediction Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                forecast_horizon = st.slider("Forecast Horizon (days)", 7, 180, 90)
                confidence_threshold = st.slider("Confidence Threshold (%)", 70, 95, 80)
            with col_b:
                auto_refill_days = st.slider("Auto-Refill Threshold (days)", 1, 14, 7)
                min_data_points = st.slider("Minimum Data Points", 1, 10, 3)
            
            if st.button("Update Prediction Model"):
                st.success("Prediction model updated with new settings!")
    
    else:
        st.info("No prediction data available yet. Start by placing some orders through the AI Assistant.")
        if st.button("Generate Demo Predictions"):
            init_demo_data()
            predictions = predict_refill_dates()
            st.rerun()

# Agent System Page
elif selected == "Agent System":
    st.title("🤖 Multi-Agent Architecture")
    
    st.markdown("""
    TrackFusion 3 uses a sophisticated multi-agent system to automate pharmacy operations. 
    Each agent has specialized responsibilities and works collaboratively to ensure seamless operation.
    """)
    
    # Agent Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    agents_info = [
        {
            "name": "Ordering Agent",
            "icon": "🛒",
            "status": st.session_state.agents["Ordering Agent"]["status"],
            "description": "Handles customer interactions, processes natural language orders, and manages cart operations.",
            "metrics": {"Requests Today": random.randint(50, 200), "Success Rate": "98.5%"}
        },
        {
            "name": "Forecast Agent",
            "icon": "📈",
            "status": st.session_state.agents["Forecast Agent"]["status"],
            "description": "Analyzes consumption patterns, predicts refill needs, and optimizes inventory levels.",
            "metrics": {"Predictions Today": random.randint(20, 100), "Accuracy": "94.2%"}
        },
        {
            "name": "Procurement Agent",
            "icon": "📦",
            "status": st.session_state.agents["Procurement Agent"]["status"],
            "description": "Automates purchase orders, negotiates with suppliers, and manages procurement workflows.",
            "metrics": {"POs Generated": random.randint(5, 30), "Cost Savings": "12.7%"}
        },
        {
            "name": "Safety Agent",
            "icon": "🛡️",
            "status": st.session_state.agents["Safety Agent"]["status"],
            "description": "Validates prescriptions, checks drug interactions, and ensures medication safety.",
            "metrics": {"Checks Today": random.randint(100, 300), "Issues Flagged": random.randint(0, 5)}
        }
    ]
    
    for idx, agent in enumerate(agents_info):
        with [col1, col2, col3, col4][idx]:
            st.markdown(f'<div class="agent-card">', unsafe_allow_html=True)
            st.markdown(f"### {agent['icon']} {agent['name']}")
            st.markdown(f"**Status:** `{agent['status']}`")
            st.markdown(f"*{agent['description']}*")
            
            for metric, value in agent["metrics"].items():
                st.write(f"**{metric}:** {value}")
            
            # Agent controls
            if st.button(f"Restart {agent['name'].split()[0]}", key=f"restart_{idx}", use_container_width=True):
                st.session_state.agents[agent["name"]]["status"] = "Active"
                st.session_state.agents[agent["name"]]["last_active"] = datetime.now()
                st.success(f"{agent['name']} restarted successfully!")
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Agent Communication Graph
    st.subheader("🔄 Agent Communication Flow")
    
    st.markdown("""
    ```mermaid
    graph LR
        A[Customer] --> B[Ordering Agent]
        B --> C[Safety Agent]
        C --> D[Forecast Agent]
        D --> E[Procurement Agent]
        E --> F[Supplier APIs]
        E --> G[Mediloon CMS]
        F --> H[Inventory Updated]
        G --> H
        H --> I[Customer Notified]
    ```
    """)
    
    # Agent Workflow Simulation
    st.subheader("⚙️ Workflow Simulation")
    
    workflow_type = st.selectbox(
        "Select Workflow Type",
        ["Standard Order Processing", "Emergency Refill", "Prescription Verification", "Supplier Negotiation"]
    )
    
    if st.button("🚀 Execute Full Workflow", type="primary", use_container_width=True):
        with st.spinner("Orchestrating multi-agent workflow..."):
            # Simulate agent coordination
            progress_bar = st.progress(0)
            
            steps = [
                "Initializing agents...",
                "Ordering Agent processing request...",
                "Safety Agent validating prescriptions...",
                "Forecast Agent analyzing patterns...",
                "Procurement Agent generating PO...",
                "Integrating with MCP tools...",
                "Updating external systems...",
                "Workflow completed!"
            ]
            
            result_container = st.container()
            
            for i, step in enumerate(steps):
                time.sleep(0.5)
                progress_bar.progress((i + 1) / len(steps))
                
                with result_container:
                    st.success(f"✅ {step}")
            
            # Show detailed log
            with st.expander("📋 View Detailed Agent Log"):
                workflow_log = simulate_agent_workflow()
                for log in workflow_log:
                    if log.strip():
                        st.code(log)
    
    # Agent Configuration
    st.subheader("⚙️ Agent Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.write("**Agent Settings**")
        
        for agent_name in st.session_state.agents.keys():
            current_status = st.session_state.agents[agent_name]["status"]
            new_status = st.selectbox(
                f"{agent_name} Status",
                ["Active", "Standby", "Maintenance", "Disabled"],
                index=["Active", "Standby", "Maintenance", "Disabled"].index(current_status),
                key=f"status_{agent_name}"
            )
            
            if new_status != current_status:
                st.session_state.agents[agent_name]["status"] = new_status
                st.session_state.agents[agent_name]["last_active"] = datetime.now()
    
    with config_col2:
        st.write("**Communication Settings**")
        
        agent_autonomy = st.slider("Agent Autonomy Level", 0, 100, 75)
        decision_threshold = st.slider("Decision Confidence Threshold", 0.5, 1.0, 0.8)
        retry_attempts = st.number_input("Max Retry Attempts", 1, 10, 3)
        
        st.write("**Inter-Agent Communication**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.checkbox("Enable Real-time Sync", value=True)
            st.checkbox("Log All Communications", value=True)
        with col_b:
            st.checkbox("Allow Override", value=False)
            st.checkbox("Failover Support", value=True)
        
        if st.button("Apply Configuration", use_container_width=True):
            st.success("Agent configuration updated successfully!")

# Inventory Page
elif selected == "Inventory":
    st.title("📦 Smart Inventory Management")
    
    # Inventory Overview
    col1, col2, col3 = st.columns(3)
    
    total_stock_value = sum(item["current_stock"] * MEDICINES[item["medicine"]]["price"] 
                           for item in st.session_state.inventory)
    
    with col1:
        st.metric("Total SKUs", len(st.session_state.inventory))
    with col2:
        st.metric("Total Stock Value", f"${total_stock_value:,.2f}")
    with col3:
        low_stock_count = sum(1 for item in st.session_state.inventory 
                             if item["current_stock"] < item["reorder_level"])
        st.metric("Low Stock Items", low_stock_count, delta="Needs attention" if low_stock_count > 0 else "Optimal")
    
    # Inventory Management Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Stock Levels", "Reorder Management", "Supplier Info", "Analytics"])
    
    with tab1:
        st.subheader("Current Stock Levels")
        
        # Interactive inventory editor
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            
            # Add calculated columns
            inv_df["value"] = inv_df.apply(
                lambda row: row["current_stock"] * MEDICINES[row["medicine"]]["price"], 
                axis=1
            )
            inv_df["status"] = inv_df.apply(
                lambda row: "⚠️ Low" if row["current_stock"] < row["reorder_level"] else "✅ Good", 
                axis=1
            )
            inv_df["days_since_order"] = inv_df["last_ordered"].apply(
                lambda x: (datetime.now() - x).days if isinstance(x, datetime) else 0
            )
            
            # Display editable dataframe
            edited_df = st.data_editor(
                inv_df[[
                    "medicine", "category", "current_stock", "reorder_level", 
                    "status", "supplier", "value", "days_since_order"
                ]],
                column_config={
                    "medicine": st.column_config.TextColumn("Medicine", width="medium"),
                    "category": st.column_config.TextColumn("Category", width="small"),
                    "current_stock": st.column_config.NumberColumn(
                        "Current Stock",
                        min_value=0,
                        max_value=1000,
                        step=1,
                        width="small"
                    ),
                    "reorder_level": st.column_config.NumberColumn(
                        "Reorder Level",
                        min_value=1,
                        max_value=500,
                        step=1,
                        width="small"
                    ),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "value": st.column_config.NumberColumn(
                        "Stock Value",
                        format="$%.2f",
                        width="small"
                    )
                },
                use_container_width=True,
                num_rows="dynamic",
                hide_index=True
            )
            
            if st.button("💾 Save Inventory Changes", use_container_width=True):
                # Update session state
                for idx, row in edited_df.iterrows():
                    if idx < len(st.session_state.inventory):
                        st.session_state.inventory[idx]["current_stock"] = row["current_stock"]
                        st.session_state.inventory[idx]["reorder_level"] = row["reorder_level"]
                
                st.success("Inventory updated successfully!")
                st.rerun()
        
        else:
            st.info("No inventory data available.")
    
    with tab2:
        st.subheader("📋 Reorder Management")
        
        # Generate reorder suggestions
        reorder_suggestions = []
        for item in st.session_state.inventory:
            if item["current_stock"] < item["reorder_level"]:
                suggestion = {
                    "Medicine": item["medicine"],
                    "Current Stock": item["current_stock"],
                    "Reorder Level": item["reorder_level"],
                    "Deficit": item["reorder_level"] - item["current_stock"],
                    "Supplier": item["supplier"],
                    "Lead Time": f"{item['lead_time_days']} days",
                    "Suggested Order": max(item["reorder_level"] * 2, 30)
                }
                reorder_suggestions.append(suggestion)
        
        if reorder_suggestions:
            st.warning(f"⚠️ {len(reorder_suggestions)} items need reordering!")
            
            reorder_df = pd.DataFrame(reorder_suggestions)
            st.dataframe(reorder_df, use_container_width=True)
            
            # Bulk reorder options
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📧 Generate Purchase Orders", use_container_width=True):
                    st.success(f"Generated POs for {len(reorder_suggestions)} items!")
                    st.info("Purchase orders sent to Procurement Agent for processing.")
            
            with col_b:
                selected_items = st.multiselect(
                    "Select items for manual reorder",
                    [s["Medicine"] for s in reorder_suggestions],
                    default=[s["Medicine"] for s in reorder_suggestions[:3]]
                )
                
                if st.button("🛒 Add to Cart", use_container_width=True):
                    quantities = []
                    for med in selected_items:
                        suggestion = next(s for s in reorder_suggestions if s["Medicine"] == med)
                        quantities.append(suggestion["Suggested Order"])
                    
                    add_to_cart(selected_items, quantities)
                    st.success(f"Added {len(selected_items)} items to cart for reorder!")
        else:
            st.success("✅ All stock levels are above reorder points. No action needed.")
    
    with tab3:
        st.subheader("🏢 Supplier Information")
        
        if st.session_state.inventory:
            # Group by supplier
            supplier_data = {}
            for item in st.session_state.inventory:
                supplier = item["supplier"]
                if supplier not in supplier_data:
                    supplier_data[supplier] = {
                        "items": [],
                        "total_value": 0,
                        "last_order": item["last_ordered"]
                    }
                
                supplier_data[supplier]["items"].append(item["medicine"])
                supplier_data[supplier]["total_value"] += item["current_stock"] * MEDICINES[item["medicine"]]["price"]
            
            # Display supplier info
            for supplier, data in supplier_data.items():
                with st.expander(f"🏭 {supplier}"):
                    st.write(f"**Items Supplied:** {len(data['items'])}")
                    st.write(f"**Total Inventory Value:** ${data['total_value']:,.2f}")
                    st.write(f"**Last Order:** {data['last_order'].strftime('%Y-%m-%d')}")
                    st.write(f"**Items:** {', '.join(data['items'][:5])}{'...' if len(data['items']) > 5 else ''}")
                    
                    # Supplier actions
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Contact {supplier}", key=f"contact_{supplier}"):
                            st.info(f"📧 Contact form opened for {supplier}")
                    with col_b:
                        if st.button(f"Order History", key=f"history_{supplier}"):
                            st.info(f"📊 Loading order history for {supplier}...")
    
    with tab4:
        st.subheader("📊 Inventory Analytics")
        
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            
            # Add value column
            inv_df["value"] = inv_df.apply(
                lambda row: row["current_stock"] * MEDICINES[row["medicine"]]["price"], 
                axis=1
            )
            
            # Category analysis
            st.write("**Stock by Category**")
            category_summary = inv_df.groupby("category").agg({
                "medicine": "count",
                "current_stock": "sum",
                "value": "sum"
            }).reset_index()
            
            category_summary.columns = ["Category", "SKU Count", "Total Stock", "Total Value"]
            st.dataframe(category_summary, use_container_width=True)
            
            # Stock distribution
            st.write("**Stock Distribution**")
            if PLOTLY_AVAILABLE:
                fig = px.pie(
                    inv_df.nlargest(10, "value"),
                    values="value",
                    names="medicine",
                    title="Top 10 Medicines by Stock Value"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(inv_df.nlargest(10, "value").set_index("medicine")["value"])
            
            # Stock aging analysis
            st.write("**Stock Aging Analysis**")
            inv_df["days_in_stock"] = inv_df["last_ordered"].apply(
                lambda x: (datetime.now() - x).days if isinstance(x, datetime) else 0
            )
            
            aging_summary = inv_df.groupby(
                pd.cut(inv_df["days_in_stock"], 
                      bins=[0, 30, 90, 180, 365, float('inf')],
                      labels=["<30 days", "30-90 days", "90-180 days", "180-365 days", ">365 days"])
            ).agg({
                "medicine": "count",
                "value": "sum"
            }).reset_index()
            
            aging_summary.columns = ["Age Category", "SKU Count", "Total Value"]
            st.dataframe(aging_summary, use_container_width=True)

# Workflow Page
elif selected == "Workflow":
    st.title("⚙️ MCP & Workflow Automation")
    
    st.markdown("""
    TrackFusion 3 integrates with Multiple Connection Platforms (MCP) to automate real-world pharmacy operations.
    The system connects AI agents with external tools and services for end-to-end automation.
    """)
    
    # MCP Integration Status
    st.subheader("🔗 MCP Integration Status")
    
    mcp_connections = {
        "Zapier": {
            "status": "✅ Connected",
            "description": "Automates email, SMS, and WhatsApp notifications",
            "last_used": "2 hours ago",
            "usage": "High"
        },
        "n8n": {
            "status": "✅ Connected",
            "description": "Triggers API calls to distributors and suppliers",
            "last_used": "1 hour ago",
            "usage": "Medium"
        },
        "Mediloon CMS": {
            "status": "✅ Connected",
            "description": "Synchronizes inventory and order data",
            "last_used": "5 minutes ago",
            "usage": "Continuous"
        },
        "Supplier APIs": {
            "status": "⚠️ Partial",
            "description": "Connects to pharmaceutical suppliers",
            "last_used": "Yesterday",
            "usage": "Medium"
        },
        "Payment Gateway": {
            "status": "✅ Connected",
            "description": "Processes customer payments",
            "last_used": "30 minutes ago",
            "usage": "High"
        }
    }
    
    # Display connection status
    cols = st.columns(len(mcp_connections))
    for idx, (name, info) in enumerate(mcp_connections.items()):
        with cols[idx]:
            status_color = "green" if "✅" in info["status"] else "orange" if "⚠️" in info["status"] else "red"
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {status_color}; background-color: #f8f9fa;">
                <h4>{name}</h4>
                <p><strong>Status:</strong> {info['status']}</p>
                <p><small>{info['description']}</small></p>
                <p><small>Last used: {info['last_used']}</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Workflow Designer
    st.subheader("🎨 Workflow Designer")
    
    workflow_type = st.selectbox(
        "Select Workflow Template",
        [
            "Standard Order Fulfillment",
            "Emergency Refill Process",
            "Prescription Verification Flow",
            "Inventory Replenishment",
            "Customer Notification Chain",
            "Custom Workflow"
        ]
    )
    
    if workflow_type == "Standard Order Fulfillment":
        st.markdown("""
        **Standard Order Fulfillment Workflow:**
        
        1. **Customer Places Order** → AI Assistant processes natural language
        2. **Safety Validation** → Safety Agent checks prescriptions
        3. **Inventory Check** → System verifies stock availability
        4. **Payment Processing** → Payment gateway integration
        5. **Order Confirmation** → Email/SMS notification via Zapier
        6. **Supplier Notification** → n8n triggers supplier API
        7. **CMS Update** → Mediloon CMS synchronized
        8. **Delivery Tracking** → Customer receives updates
        
        **Estimated Time:** 2-5 minutes
        **Success Rate:** 99.2%
        """)
    
    # Workflow Execution
    st.subheader("🚀 Execute Workflow")
    
    execution_tab1, execution_tab2 = st.tabs(["Manual Execution", "Scheduled Automation"])
    
    with execution_tab1:
        st.write("Execute workflow manually with custom parameters")
        
        workflow_params = {}
        
        col1, col2 = st.columns(2)
        with col1:
            workflow_params["priority"] = st.selectbox("Priority", ["Normal", "High", "Emergency"])
            workflow_params["notify_customer"] = st.checkbox("Notify Customer", value=True)
            workflow_params["update_inventory"] = st.checkbox("Update Inventory", value=True)
        
        with col2:
            workflow_params["payment_required"] = st.checkbox("Payment Required", value=True)
            workflow_params["supplier_notify"] = st.checkbox("Notify Supplier", value=True)
            workflow_params["generate_reports"] = st.checkbox("Generate Reports", value=True)
        
        if st.button("▶️ Execute Workflow Now", type="primary", use_container_width=True):
            with st.spinner(f"Executing {workflow_type}..."):
                # Simulate workflow execution
                progress_bar = st.progress(0)
                status_container = st.empty()
                
                execution_steps = [
                    "Initializing workflow engine...",
                    "Connecting to MCP tools...",
                    "Executing agent coordination...",
                    "Processing external integrations...",
                    "Updating all systems...",
                    "Workflow execution complete!"
                ]
                
                for i, step in enumerate(execution_steps):
                    time.sleep(0.7)
                    progress_bar.progress((i + 1) / len(execution_steps))
                    status_container.info(f"⏳ Step {i+1}/{len(execution_steps)}: {step}")
                
                # Show results
                st.success("✅ Workflow executed successfully!")
                
                # Generate execution report
                with st.expander("📊 Execution Report"):
                    st.write("**Workflow Summary**")
                    st.write(f"- **Type:** {workflow_type}")
                    st.write(f"- **Priority:** {workflow_params['priority']}")
                    st.write(f"- **Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- **Duration:** {len(execution_steps) * 0.7:.1f} seconds")
                    st.write(f"- **Status:** Completed Successfully")
                    
                    st.write("**MCP Integrations Used:**")
                    for name, info in mcp_connections.items():
                        if info["usage"] in ["High", "Medium", "Continuous"]:
                            st.write(f"- {name}: {info['status']}")
                
                # Trigger agent workflow
                st.info("🤖 Activating Multi-Agent System...")
                workflow_log = simulate_agent_workflow()
                for log in workflow_log:
                    st.code(log)
    
    with execution_tab2:
        st.write("Schedule automated workflow execution")
        
        schedule_type = st.radio(
            "Schedule Type",
            ["Daily", "Weekly", "Monthly", "Custom Cron"]
        )
        
        if schedule_type == "Daily":
            schedule_time = st.time_input("Execution Time", value=datetime.now().time())
        elif schedule_type == "Weekly":
            col_a, col_b = st.columns(2)
            with col_a:
                schedule_day = st.selectbox("Day of Week", 
                                          ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            with col_b:
                schedule_time = st.time_input("Execution Time", value=datetime.now().time())
        elif schedule_type == "Monthly":
            schedule_day = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
            schedule_time = st.time_input("Execution Time", value=datetime.now().time())
        else:
            cron_expression = st.text_input("Cron Expression", value="0 9 * * *")
            st.caption("Example: 0 9 * * * = Daily at 9:00 AM")
        
        enable_schedule = st.checkbox("Enable Schedule", value=True)
        
        if st.button("💾 Save Schedule", use_container_width=True):
            if enable_schedule:
                st.success(f"✅ Workflow scheduled for {schedule_type} execution!")
                st.info("Scheduled workflows will run automatically in the background.")
            else:
                st.warning("⚠️ Schedule saved but disabled. Enable to activate.")

# Settings Page
elif selected == "Settings":
    st.title("⚙️ System Configuration")
    
    settings_tabs = st.tabs(["General", "AI & Agents", "Notifications", "API Keys", "System Info"])
    
    with settings_tabs[0]:  # General
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Application Settings**")
            
            app_language = st.selectbox(
                "Interface Language",
                ["English", "German", "Arabic", "Spanish", "French"],
                index=0
            )
            
            timezone = st.selectbox(
                "Timezone",
                ["UTC", "EST", "PST", "CET", "IST", "AEST"],
                index=0
            )
            
            date_format = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY", "Month DD, YYYY"],
                index=0
            )
            
            currency = st.selectbox(
                "Currency",
                ["USD", "EUR", "GBP", "AED", "INR"],
                index=0
            )
        
        with col2:
            st.write("**Feature Toggles**")
            
            enable_voice = st.checkbox("Enable Voice Input", value=True)
            enable_predictions = st.checkbox("Enable Predictive Analytics", value=True)
            enable_auto_refill = st.checkbox("Enable Auto-Refill", value=True)
            enable_agent_system = st.checkbox("Enable Agent System", value=True)
            
            st.write("**Data Retention**")
            retention_days = st.slider("Retain Data For (days)", 30, 365, 90)
            auto_backup = st.checkbox("Automatic Daily Backup", value=True)
        
        if st.button("💾 Save General Settings", use_container_width=True):
            st.success("General settings saved successfully!")
    
    with settings_tabs[1]:  # AI & Agents
        st.subheader("AI & Agent Configuration")
        
        st.write("**AI Model Settings**")
        
        ai_provider = st.radio(
            "AI Provider",
            ["OpenAI GPT-4", "Anthropic Claude", "Local LLM", "Hybrid Mode"],
            horizontal=True
        )
        
        if ai_provider == "OpenAI GPT-4":
            temperature = st.slider("Model Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)
        elif ai_provider == "Local LLM":
            model_name = st.selectbox(
                "Local Model",
                ["Llama 2 7B", "Mistral 7B", "Phi-2", "Custom Model"]
            )
            gpu_acceleration = st.checkbox("GPU Acceleration", value=True)
        
        st.write("**Agent Configuration**")
        
        agent_autonomy = st.slider("Agent Autonomy Level", 0, 100, 75,
                                 help="Higher values allow agents to make more decisions without human confirmation")
        
        decision_timeout = st.number_input("Decision Timeout (seconds)", 1, 300, 30)
        
        col_a, col_b = st.columns(2)
        with col_a:
            enable_agent_logging = st.checkbox("Enable Agent Logging", value=True)
            enable_agent_learning = st.checkbox("Enable Agent Learning", value=True)
        with col_b:
            allow_agent_override = st.checkbox("Allow Human Override", value=True)
            enable_failover = st.checkbox("Enable Failover Mode", value=True)
        
        if st.button("💾 Save AI Settings", use_container_width=True):
            st.success("AI and Agent settings saved successfully!")
    
    with settings_tabs[2]:  # Notifications
        st.subheader("Notification Settings")
        
        st.write("**Notification Channels**")
        
        notification_cols = st.columns(3)
        with notification_cols[0]:
            st.checkbox("Email Notifications", value=True)
            st.checkbox("Order Confirmations", value=True)
            st.checkbox("Shipping Updates", value=True)
        
        with notification_cols[1]:
            st.checkbox("SMS Notifications", value=True)
            st.checkbox("Refill Reminders", value=True)
            st.checkbox("Prescription Alerts", value=True)
        
        with notification_cols[2]:
            st.checkbox("WhatsApp Messages", value=False)
            st.checkbox("Inventory Alerts", value=True)
            st.checkbox("System Notifications", value=True)
        
        st.write("**Notification Preferences**")
        
        notify_immediate = st.checkbox("Immediate Notifications", value=True)
        notify_daily = st.checkbox("Daily Summary", value=False)
        notify_weekly = st.checkbox("Weekly Report", value=True)
        
        quiet_hours = st.checkbox("Enable Quiet Hours", value=False)
        if quiet_hours:
            quiet_start, quiet_end = st.slider(
                "Quiet Hours",
                value=(22, 8),
                format="%H:00"
            )
        
        if st.button("💾 Save Notification Settings", use_container_width=True):
            st.success("Notification settings saved successfully!")
    
    with settings_tabs[3]:  # API Keys
        st.subheader("API Integration Keys")
        
        st.warning("⚠️ API keys are sensitive information. Store them securely.")
        
        zapier_key = st.text_input("Zapier API Key", type="password")
        n8n_webhook = st.text_input("n8n Webhook URL", type="password")
        mediloon_api = st.text_input("Mediloon CMS API Key", type="password")
        
        # Provider-specific keys
        ai_provider_keys = st.expander("AI Provider API Keys")
        with ai_provider_keys:
            openai_key = st.text_input("OpenAI API Key", type="password")
            anthropic_key = st.text_input("Anthropic API Key", type="password")
            huggingface_key = st.text_input("HuggingFace API Key", type="password")
        
        # Test connections
        if st.button("🔗 Test API Connections", use_container_width=True):
            with st.spinner("Testing API connections..."):
                time.sleep(2)
                
                test_results = []
                
                if zapier_key:
                    test_results.append(("Zapier", "✅ Connected", "green"))
                else:
                    test_results.append(("Zapier", "❌ Not Configured", "red"))
                
                if n8n_webhook:
                    test_results.append(("n8n", "✅ Connected", "green"))
                else:
                    test_results.append(("n8n", "❌ Not Configured", "red"))
                
                if mediloon_api:
                    test_results.append(("Mediloon CMS", "✅ Connected", "green"))
                else:
                    test_results.append(("Mediloon CMS", "❌ Not Configured", "red"))
                
                # Display results
                for name, status, color in test_results:
                    st.markdown(f"- **{name}:** <span style='color:{color}'>{status}</span>", 
                              unsafe_allow_html=True)
        
        if st.button("💾 Save API Keys", use_container_width=True):
            st.success("API keys saved successfully!")
            st.info("Keys are stored in session memory. For production, use environment variables or a secure vault.")
    
    with settings_tabs[4]:  # System Info
        st.subheader("System Information")
        
        # System metrics
        st.write("**Application Information**")
        
        info_cols = st.columns(3)
        with info_cols[0]:
            st.write("**Version:** 3.0.1")
            st.write("**Build Date:** 2024-01-15")
            st.write("**License:** MIT")
        
        with info_cols[1]:
            st.write("**Python Version:** 3.9+")
            st.write("**Streamlit:** 1.28.0+")
            st.write("**Environment:** Development")
        
        with info_cols[2]:
            st.write("**Database:** In-Memory")
            st.write("**AI Models:** Simulated")
            st.write("**Agents:** 4 Active")
        
        # System status
        st.write("**System Status**")
        
        status_items = [
            ("AI Assistant", "✅ Operational", "green"),
            ("Predictive Engine", "✅ Operational", "green"),
            ("Agent System", "✅ Operational", "green"),
            ("Inventory Management", "✅ Operational", "green"),
            ("Workflow Automation", "⚠️ Limited", "orange"),
            ("External APIs", "✅ Connected", "green")
        ]
        
        for name, status, color in status_items:
            st.markdown(f"- **{name}:** <span style='color:{color}'>{status}</span>", 
                      unsafe_allow_html=True)
        
        # System actions
        st.write("**System Actions**")
        
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.session_state.clear()
                st.success("Cache cleared! Reinitializing demo data...")
                init_demo_data()
                st.rerun()
        
        with action_cols[1]:
            if st.button("📊 Generate System Report", use_container_width=True):
                st.success("System report generated!")
                st.download_button(
                    label="📥 Download Report",
                    data=json.dumps({
                        "orders_count": len(st.session_state.orders),
                        "inventory_count": len(st.session_state.inventory),
                        "agents_status": st.session_state.agents,
                        "timestamp": datetime.now().isoformat()
                    }, indent=2),
                    file_name="system_report.json",
                    mime="application/json"
                )
        
        with action_cols[2]:
            if st.button("🔄 Restart System", use_container_width=True):
                st.warning("System restart initiated...")
                time.sleep(1)
                st.success("System restarted successfully!")
                st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 HackFusion Project")
st.sidebar.markdown("**TrackFusion 3**")
st.sidebar.markdown("*AI-Driven Autonomous Pharmacy*")
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions:**")

if st.sidebar.button("🔄 Reset Demo Data", use_container_width=True):
    st.session_state.clear()
    init_demo_data()
    st.success("Demo data reset!")
    st.rerun()

if st.sidebar.button("📊 Export All Data", use_container_width=True):
    all_data = {
        "orders": st.session_state.orders,
        "inventory": st.session_state.inventory,
        "predictions": st.session_state.predictions,
        "user_profile": st.session_state.user_profile
    }
    
    st.sidebar.download_button(
        label="📥 Download Data",
        data=json.dumps(all_data, indent=2, default=str),
        file_name="trackfusion_data.json",
        mime="application/json"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ for HackFusion")
st.sidebar.caption(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize on first run
if __name__ == "__main__":
    # Ensure demo data is initialized
    if not st.session_state.orders:
        init_demo_data()
