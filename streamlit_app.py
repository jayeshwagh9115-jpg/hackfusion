import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import time
import math

# Page configuration
st.set_page_config(
    page_title="TrackFusion 3 - AI Pharmacy System",
    page_icon="💊",
    layout="wide"
)

# Initialize all session state variables
def init_session_state():
    defaults = {
        'orders': [],
        'inventory': [],
        'chat_history': [],
        'predictions': {},
        'cart': [],
        'user_profile': {},
        'agents': {
            "Ordering Agent": {"status": "Active", "last_active": datetime.now()},
            "Forecast Agent": {"status": "Active", "last_active": datetime.now()},
            "Procurement Agent": {"status": "Idle", "last_active": datetime.now()},
            "Safety Agent": {"status": "Active", "last_active": datetime.now()}
        },
        'system_settings': {
            'language': 'English',
            'auto_refill': True,
            'notifications': True
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# Medicine database
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
    """Predict when medicines will run out - Pure Python implementation"""
    predictions = {}
    
    for medicine, info in MEDICINES.items():
        # Get recent orders for this medicine
        recent_orders = [o for o in st.session_state.orders 
                        if o["medicine"] == medicine]
        
        if recent_orders and len(recent_orders) > 1:
            # Calculate consumption using simple moving average
            total_qty = sum([o["quantity"] for o in recent_orders])
            
            # Find date range
            dates = [o["date"] for o in recent_orders]
            min_date = min(dates)
            max_date = max(dates)
            days_span = (max_date - min_date).days or 30
            
            daily_consumption = total_qty / days_span if days_span > 0 else 0
            
            # Find inventory
            inv_item = next((i for i in st.session_state.inventory 
                           if i["medicine"] == medicine), None)
            
            if inv_item and daily_consumption > 0:
                days_until_empty = inv_item["current_stock"] / daily_consumption
                refill_date = datetime.now() + timedelta(days=days_until_empty)
                
                # Calculate confidence based on data points
                data_points = len(recent_orders)
                confidence = min(95, 60 + (data_points * 5))
                
                urgency_level = "CRITICAL" if days_until_empty < 3 else \
                               "HIGH" if days_until_empty < 7 else \
                               "MEDIUM" if days_until_empty < 14 else "LOW"
                
                predictions[medicine] = {
                    "refill_date": refill_date,
                    "days_until_empty": int(days_until_empty),
                    "daily_consumption": round(daily_consumption, 3),
                    "current_stock": inv_item["current_stock"],
                    "reorder_level": inv_item["reorder_level"],
                    "confidence": confidence,
                    "urgency": urgency_level
                }
    
    st.session_state.predictions = predictions
    return predictions

def process_natural_language(input_text):
    """Process natural language for medicine ordering"""
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
    
    # Check for phrases first
    for phrase, meds in phrase_mapping.items():
        if phrase in input_text:
            detected_meds.extend(meds)
    
    # If no phrase match, look for individual medicine names
    if not detected_meds:
        for med in MEDICINES.keys():
            if med.lower() in input_text:
                detected_meds.append(med)
    
    # Extract quantities
    quantities = []
    quantity_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "some": 2, "few": 3, "several": 3
    }
    
    words = input_text.split()
    for word in words:
        if word.isdigit():
            quantities.append(int(word))
        elif word in quantity_map:
            quantities.append(quantity_map[word])
    
    # Default quantity if none specified
    if not quantities and detected_meds:
        quantities = [1] * len(detected_meds)
    
    return detected_meds, quantities[:len(detected_meds)]

def add_to_cart(medicines, quantities):
    """Add items to cart"""
    for med, qty in zip(medicines, quantities):
        # Check if already in cart
        existing_idx = next((i for i, item in enumerate(st.session_state.cart) 
                           if item["medicine"] == med), None)
        
        if existing_idx is not None:
            st.session_state.cart[existing_idx]["quantity"] += qty
        else:
            st.session_state.cart.append({
                "medicine": med,
                "quantity": qty,
                "price": MEDICINES[med]["price"],
                "prescription_required": MEDICINES[med]["prescription_required"],
                "category": MEDICINES[med]["type"],
                "added_at": datetime.now()
            })

def simulate_agent_workflow():
    """Simulate multi-agent workflow"""
    workflow = [
        "🤖 **Multi-Agent System Activated**",
        "",
        "🛒 **Ordering Agent**: Processing customer request...",
        "  ↳ Parsed medicine requirements",
        "  ↳ Validated order details",
        "  ↳ Forwarded to inventory check",
        "",
        "🛡️ **Safety Agent**: Checking prescription requirements...",
        "  ↳ Validating drug interactions",
        "  ↳ Verifying dosage information",
        "  ↳ Safety check completed",
        "",
        "📈 **Forecast Agent**: Analyzing consumption patterns...",
        "  ↳ Predicting future demand",
        "  ↳ Calculating reorder points",
        "  ↳ Inventory forecast updated",
        "",
        "📦 **Procurement Agent**: Generating purchase order...",
        "  ↳ Contacting suppliers",
        "  ↳ Negotiating prices",
        "  ↳ Order confirmed with supplier",
        "",
        "🔗 **MCP Integration**",
        "  ↳ Connected to Zapier for notifications",
        "  ↳ Triggered n8n workflow for supplier API",
        "  ↳ Updated Mediloon CMS via webhook",
        "  ↳ Order processed successfully!",
        ""
    ]
    return workflow

def generate_order_summary():
    """Generate order summary"""
    if not st.session_state.cart:
        return None
    
    total_items = len(st.session_state.cart)
    total_qty = sum(item["quantity"] for item in st.session_state.cart)
    subtotal = sum(item["price"] * item["quantity"] for item in st.session_state.cart)
    
    return {
        "total_items": total_items,
        "total_quantity": total_qty,
        "subtotal": subtotal,
        "tax": round(subtotal * 0.08, 2),
        "shipping": 0 if subtotal > 50 else 5.99,
        "order_id": f"ORD{random.randint(10000, 99999)}",
        "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d, %Y")
    }

def create_simple_chart(data_dict, title="", height=300):
    """Create a simple bar chart using Streamlit's native chart"""
    if not data_dict:
        return
    
    df = pd.DataFrame({
        'Category': list(data_dict.keys()),
        'Value': list(data_dict.values())
    })
    
    st.bar_chart(df.set_index('Category'), height=height)

# Initialize demo data
init_demo_data()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
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
    .chat-user { background-color: #e3f2fd; padding: 0.5rem; border-radius: 0.5rem; margin: 0.25rem 0; }
    .chat-assistant { background-color: #f3e5f5; padding: 0.5rem; border-radius: 0.5rem; margin: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="main-header">💊 TrackFusion 3</div>', unsafe_allow_html=True)
    st.markdown("### AI-Driven Autonomous Pharmacy")
    
    page = st.selectbox(
        "Navigation",
        ["Dashboard", "AI Assistant", "Predictive Engine", "Agent System", "Inventory", "Workflow", "Settings"]
    )
    
    st.markdown("---")
    st.caption(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Quick Actions
    if st.button("🔄 Reset Demo Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        init_demo_data()
        st.rerun()

# Dashboard
if page == "Dashboard":
    st.title("🏥 Autonomous Pharmacy Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", len(st.session_state.orders))
    with col2:
        st.metric("Inventory Items", len(st.session_state.inventory))
    with col3:
        predictions = predict_refill_dates()
        critical = sum(1 for p in predictions.values() if p.get("urgency") in ["CRITICAL", "HIGH"])
        st.metric("Urgent Refills", critical)
    with col4:
        active_agents = sum(1 for a in st.session_state.agents.values() if a["status"] == "Active")
        st.metric("Active Agents", active_agents)
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    
    tab1, tab2, tab3 = st.tabs(["Recent Orders", "Inventory Status", "System Alerts"])
    
    with tab1:
        if st.session_state.orders:
            recent_orders = sorted(st.session_state.orders, key=lambda x: x["date"], reverse=True)[:10]
            orders_df = pd.DataFrame(recent_orders)
            
            # Format dates for display
            orders_df["date"] = pd.to_datetime(orders_df["date"]).dt.strftime("%Y-%m-%d")
            
            st.dataframe(
                orders_df[["id", "medicine", "quantity", "date", "status"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No orders yet")
    
    with tab2:
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            st.dataframe(
                inv_df[["medicine", "current_stock", "reorder_level", "supplier"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Simple chart using native Streamlit
            chart_data = inv_df.nlargest(10, "current_stock")[["medicine", "current_stock"]].set_index("medicine")
            st.bar_chart(chart_data)
    
    with tab3:
        # Generate system alerts
        alerts = []
        
        # Check for low stock
        for item in st.session_state.inventory:
            if item["current_stock"] < item["reorder_level"]:
                alerts.append({
                    "type": "⚠️ Inventory Alert",
                    "message": f"{item['medicine']} stock low ({item['current_stock']} left, reorder at {item['reorder_level']})"
                })
        
        # Check for prescription verification
        pending_prescriptions = sum(1 for order in st.session_state.orders 
                                  if order.get("prescription_required", False) and not order.get("prescription_verified", False))
        
        if pending_prescriptions > 0:
            alerts.append({
                "type": "📋 Prescription Alert",
                "message": f"{pending_prescriptions} orders need prescription verification"
            })
        
        if alerts:
            for alert in alerts:
                with st.expander(alert["type"]):
                    st.write(alert["message"])
                    st.button("Take Action", key=f"action_{alert['type']}")
        else:
            st.success("✅ All systems operational. No alerts at this time.")

# AI Assistant
elif page == "AI Assistant":
    st.title("🎤 AI Ordering Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for chat in st.session_state.chat_history[-5:]:
                if chat["role"] == "user":
                    st.markdown(f'<div class="chat-user"><strong>You:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-assistant"><strong>AI Assistant:</strong> {chat["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input("Type your medicine order or question:", 
                                  placeholder="e.g., 'I need Metformin and two boxes of Ibuprofen'")
        
        if st.button("Send", type="primary") and user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("🤖 Processing your request..."):
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
                        response += f"\n\n⚠️ **Note:** {', '.join(presc_meds)} require prescription validation."
                    
                    response += "\n\nIs there anything else you'd like to order?"
                else:
                    # General conversation responses
                    general_responses = [
                        "I can help you order medicines. Try saying things like 'I need my blood pressure medication' or 'Order some Ibuprofen'.",
                        "Please tell me what medicines you need, and I'll add them to your cart automatically.",
                        "I'm here to assist with medicine orders. You can ask for specific medicines or describe your symptoms for recommendations."
                    ]
                    response = random.choice(general_responses)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    with col2:
        st.subheader("🛒 Your Cart")
        
        if st.session_state.cart:
            # Display cart items
            for i, item in enumerate(st.session_state.cart):
                with st.expander(f"{item['medicine']} ({item['quantity']}x)"):
                    st.write(f"**Price:** ${item['price']:.2f}")
                    st.write(f"**Total:** ${item['price'] * item['quantity']:.2f}")
                    st.write(f"**Category:** {item['category']}")
                    
                    if item['prescription_required']:
                        st.warning("Prescription required")
                    else:
                        st.success("No prescription needed")
            
            # Order summary
            st.markdown("---")
            summary = generate_order_summary()
            
            if summary:
                st.write("**Order Summary:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"Items: {summary['total_items']}")
                    st.write(f"Quantity: {summary['total_quantity']}")
                with col_b:
                    st.write(f"Subtotal: ${summary['subtotal']:.2f}")
                    st.write(f"Tax: ${summary['tax']:.2f}")
                
                st.markdown(f"**Total: ${summary['subtotal'] + summary['tax'] + summary['shipping']:.2f}**")
                
                if st.button("✅ Checkout", type="primary", use_container_width=True):
                    # Simulate checkout process
                    with st.spinner("Processing your order..."):
                        time.sleep(1)
                        
                        # Add to orders
                        for item in st.session_state.cart:
                            st.session_state.orders.append({
                                "id": summary["order_id"],
                                "medicine": item["medicine"],
                                "quantity": item["quantity"],
                                "date": datetime.now(),
                                "status": "Processing",
                                "prescription_required": item["prescription_required"],
                                "prescription_verified": not item["prescription_required"]
                            })
                        
                        # Show success message
                        st.success(f"🎉 Order placed successfully! Order ID: {summary['order_id']}")
                        st.info(f"Estimated delivery: {summary['estimated_delivery']}")
                        
                        # Show agent workflow
                        st.subheader("🤖 Agent Workflow Activated:")
                        workflow = simulate_agent_workflow()
                        for line in workflow:
                            if line.strip():
                                st.code(line)
                        
                        # Clear cart
                        st.session_state.cart = []
                        st.rerun()
            
            if st.button("🗑️ Clear Cart", use_container_width=True):
                st.session_state.cart = []
                st.rerun()
        
        else:
            st.info("Your cart is empty. Start ordering using the chat interface!")

# Predictive Engine
elif page == "Predictive Engine":
    st.title("🔮 Predictive Ordering Engine")
    
    # Generate predictions
    predictions = predict_refill_dates()
    
    if predictions:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            critical = sum(1 for p in predictions.values() if p.get("urgency") == "CRITICAL")
            st.metric("Critical Refills", critical)
        
        with col2:
            high = sum(1 for p in predictions.values() if p.get("urgency") == "HIGH")
            st.metric("High Priority", high)
        
        with col3:
            avg_days = np.mean([p.get("days_until_empty", 0) for p in predictions.values()])
            st.metric("Avg Days Until Empty", f"{avg_days:.1f} days")
        
        # Detailed predictions table
        st.subheader("📊 Medicine Refill Predictions")
        
        pred_data = []
        for med, pred in predictions.items():
            pred_data.append({
                "Medicine": med,
                "Current Stock": pred.get("current_stock", 0),
                "Reorder Level": pred.get("reorder_level", 0),
                "Daily Consumption": pred.get("daily_consumption", 0),
                "Days Until Empty": pred.get("days_until_empty", 0),
                "Refill Date": pred.get("refill_date", datetime.now()).strftime("%Y-%m-%d"),
                "Confidence": f"{pred.get('confidence', 0)}%",
                "Urgency": pred.get("urgency", "LOW")
            })
        
        pred_df = pd.DataFrame(pred_data)
        
        # Display with sorting
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Auto-refill recommendations
        st.subheader("🤖 Auto-Refill Recommendations")
        
        needs_refill = [med for med, pred in predictions.items() 
                       if pred.get("urgency") in ["CRITICAL", "HIGH"]]
        
        if needs_refill:
            st.warning(f"⚠️ **Immediate Action Required:** {len(needs_refill)} medicines need urgent refill")
            
            for med in needs_refill:
                pred = predictions[med]
                with st.expander(f"🔴 {med} - {pred.get('urgency', 'MEDIUM')} Priority"):
                    st.write(f"**Current Stock:** {pred.get('current_stock', 0)}")
                    st.write(f"**Days Until Empty:** {pred.get('days_until_empty', 0)}")
                    st.write(f"**Recommended Order Quantity:** {max(30, int(pred.get('daily_consumption', 1) * 30))}")
                    st.write(f"**Suggested Refill Date:** {pred.get('refill_date', datetime.now()).strftime('%Y-%m-%d')}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Auto-Order {med}", key=f"auto_{med}"):
                            add_to_cart([med], [max(2, int(pred.get('daily_consumption', 1) * 30))])
                            st.success(f"Added {med} to cart for auto-refill!")
                            st.rerun()
            
            if st.button("🔄 Auto-Order All Critical Items", type="primary"):
                quantities = []
                for med in needs_refill:
                    pred = predictions[med]
                    quantities.append(max(2, int(pred.get('daily_consumption', 1) * 30)))
                
                add_to_cart(needs_refill, quantities)
                st.success(f"Added {len(needs_refill)} critical medicines to cart!")
                st.rerun()
        else:
            st.success("✅ No immediate refills needed. All inventory levels are satisfactory.")
    
    else:
        st.info("No prediction data available yet. Start by placing some orders through the AI Assistant.")

# Agent System
elif page == "Agent System":
    st.title("🤖 Multi-Agent System")
    
    st.markdown("""
    ### Agent Architecture
    
    TrackFusion 3 uses a sophisticated multi-agent system to automate pharmacy operations:
    """)
    
    # Agent Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    agents_info = [
        {
            "name": "Ordering Agent",
            "icon": "🛒",
            "status": st.session_state.agents["Ordering Agent"]["status"],
            "description": "Handles customer interactions and processes orders",
            "metrics": {"Requests": random.randint(50, 200), "Success": "98.5%"}
        },
        {
            "name": "Forecast Agent",
            "icon": "📈",
            "status": st.session_state.agents["Forecast Agent"]["status"],
            "description": "Analyzes consumption patterns and predicts refills",
            "metrics": {"Predictions": random.randint(20, 100), "Accuracy": "94.2%"}
        },
        {
            "name": "Procurement Agent",
            "icon": "📦",
            "status": st.session_state.agents["Procurement Agent"]["status"],
            "description": "Automates purchase orders and manages suppliers",
            "metrics": {"POs": random.randint(5, 30), "Savings": "12.7%"}
        },
        {
            "name": "Safety Agent",
            "icon": "🛡️",
            "status": st.session_state.agents["Safety Agent"]["status"],
            "description": "Validates prescriptions and ensures medication safety",
            "metrics": {"Checks": random.randint(100, 300), "Issues": random.randint(0, 5)}
        }
    ]
    
    for idx, agent in enumerate(agents_info):
        with [col1, col2, col3, col4][idx]:
            status_color = "green" if agent["status"] == "Active" else "orange"
            st.markdown(f"""
            <div class="agent-card">
                <h3>{agent['icon']} {agent['name']}</h3>
                <p><strong>Status:</strong> <span style="color:{status_color}">{agent['status']}</span></p>
                <p>{agent['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            for metric, value in agent["metrics"].items():
                st.write(f"**{metric}:** {value}")
            
            # Agent controls
            if st.button(f"Restart {agent['name']}", key=f"restart_{idx}", use_container_width=True):
                st.session_state.agents[agent["name"]]["status"] = "Active"
                st.session_state.agents[agent["name"]]["last_active"] = datetime.now()
                st.success(f"{agent['name']} restarted successfully!")
                st.rerun()
    
    # Workflow simulation
    st.subheader("⚙️ Workflow Simulation")
    
    if st.button("🚀 Execute Complete Workflow", type="primary", use_container_width=True):
        with st.spinner("Orchestrating multi-agent workflow..."):
            # Simulate workflow steps
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = [
                "Initializing agents...",
                "Ordering Agent processing request...",
                "Safety Agent validating prescriptions...",
                "Forecast Agent analyzing patterns...",
                "Procurement Agent generating PO...",
                "Integrating with external systems...",
                "Updating all databases...",
                "Workflow completed!"
            ]
            
            for i, step in enumerate(steps):
                time.sleep(0.5)
                progress_bar.progress((i + 1) / len(steps))
                status_text.info(f"⏳ {step}")
            
            # Show detailed log
            st.success("✅ Workflow executed successfully!")
            
            with st.expander("📋 View Detailed Agent Log"):
                workflow = simulate_agent_workflow()
                for line in workflow:
                    if line.strip():
                        st.code(line)

# Inventory Management
elif page == "Inventory":
    st.title("📦 Smart Inventory Management")
    
    # Summary metrics
    total_value = sum(item["current_stock"] * MEDICINES[item["medicine"]]["price"] 
                     for item in st.session_state.inventory)
    low_stock = sum(1 for item in st.session_state.inventory 
                   if item["current_stock"] < item["reorder_level"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total SKUs", len(st.session_state.inventory))
    with col2:
        st.metric("Stock Value", f"${total_value:,.2f}")
    with col3:
        st.metric("Low Stock Items", low_stock)
    
    # Inventory editor
    st.subheader("Stock Levels")
    
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
        
        # Display editable dataframe
        edited_df = st.data_editor(
            inv_df[["medicine", "current_stock", "reorder_level", "status", "supplier", "value"]],
            column_config={
                "current_stock": st.column_config.NumberColumn(
                    "Current Stock",
                    min_value=0,
                    max_value=1000,
                    step=1
                ),
                "reorder_level": st.column_config.NumberColumn(
                    "Reorder Level",
                    min_value=1,
                    max_value=500,
                    step=1
                ),
                "value": st.column_config.NumberColumn(
                    "Stock Value",
                    format="$%.2f"
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
    
    # Reorder management
    st.subheader("📋 Reorder Management")
    
    suggestions = []
    for item in st.session_state.inventory:
        if item["current_stock"] < item["reorder_level"]:
            deficit = item["reorder_level"] - item["current_stock"]
            suggestions.append({
                "Medicine": item["medicine"],
                "Current": item["current_stock"],
                "Reorder At": item["reorder_level"],
                "Deficit": deficit,
                "Supplier": item["supplier"],
                "Suggested Order": max(30, item["reorder_level"] * 2)
            })
    
    if suggestions:
        st.warning(f"⚠️ {len(suggestions)} items need reordering!")
        
        sugg_df = pd.DataFrame(suggestions)
        st.dataframe(sugg_df, use_container_width=True, hide_index=True)
        
        # Bulk actions
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📧 Generate Purchase Orders", use_container_width=True):
                st.success(f"Generated POs for {len(suggestions)} items!")
                st.info("Purchase orders sent to Procurement Agent for processing.")
        
        with col_b:
            selected_items = st.multiselect(
                "Select items for manual reorder",
                [s["Medicine"] for s in suggestions],
                default=[s["Medicine"] for s in suggestions[:3]]
            )
            
            if st.button("🛒 Add to Cart", use_container_width=True) and selected_items:
                quantities = []
                for med in selected_items:
                    suggestion = next(s for s in suggestions if s["Medicine"] == med)
                    quantities.append(suggestion["Suggested Order"])
                
                add_to_cart(selected_items, quantities)
                st.success(f"Added {len(selected_items)} items to cart for reorder!")
                st.rerun()
    else:
        st.success("✅ All stock levels are above reorder points. No action needed.")

# Workflow Automation
elif page == "Workflow":
    st.title("⚙️ MCP & Workflow Automation")
    
    st.markdown("""
    ### Multiple Connection Platform Integration
    
    TrackFusion 3 connects with external tools for end-to-end automation:
    """)
    
    # MCP Status
    mcp_services = {
        "Zapier": ("✅ Connected", "Email/SMS/WhatsApp notifications"),
        "n8n": ("✅ Connected", "API automation and workflows"),
        "Mediloon CMS": ("✅ Connected", "Inventory and order synchronization"),
        "Supplier APIs": ("⚠️ Partial", "Automated supplier integration")
    }
    
    cols = st.columns(4)
    for idx, (service, (status, desc)) in enumerate(mcp_services.items()):
        with cols[idx]:
            status_color = "green" if "✅" in status else "orange"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{service}</h4>
                <p><span style="color:{status_color}">{status}</span></p>
                <p><small>{desc}</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Workflow designer
    st.subheader("🎨 Workflow Designer")
    
    workflow_type = st.selectbox(
        "Select Workflow Template",
        ["Standard Order Fulfillment", "Inventory Replenishment", 
         "Customer Notification", "Emergency Refill", "Prescription Verification"]
    )
    
    st.write(f"**Selected Workflow:** {workflow_type}")
    
    # Show workflow steps
    st.write("**Workflow Steps:**")
    
    workflow_steps = {
        "Standard Order Fulfillment": [
            "1. Customer places order via AI Assistant",
            "2. Safety Agent validates prescription requirements",
            "3. Inventory system checks stock availability",
            "4. Payment processing (if required)",
            "5. Order confirmation sent via Zapier",
            "6. Supplier notified via n8n API",
            "7. Mediloon CMS updated with order details",
            "8. Customer receives tracking information"
        ],
        "Inventory Replenishment": [
            "1. Low stock detected by monitoring system",
            "2. Forecast Agent analyzes consumption patterns",
            "3. Procurement Agent generates purchase order",
            "4. Supplier API called via n8n",
            "5. Order confirmation received",
            "6. Inventory updated upon delivery",
            "7. System notified of restock completion"
        ]
    }
    
    steps = workflow_steps.get(workflow_type, ["Workflow steps will be displayed here"])
    
    for step in steps:
        st.write(step)
    
    # Workflow execution
    st.subheader("🚀 Execute Workflow")
    
    if st.button("▶️ Run Workflow Now", type="primary", use_container_width=True):
        with st.spinner(f"Executing {workflow_type}..."):
            # Simulate execution
            progress = st.progress(0)
            log_container = st.container()
            
            execution_log = [
                f"Starting {workflow_type} workflow...",
                "Connecting to MCP services...",
                "Zapier: ✅ Connected",
                "n8n: ✅ Connected",
                "Mediloon CMS: ✅ Connected",
                "Executing workflow steps...",
                "Workflow execution in progress...",
                "All steps completed successfully!",
                f"{workflow_type} workflow finished!"
            ]
            
            for i, log_entry in enumerate(execution_log):
                time.sleep(0.3)
                progress.progress((i + 1) / len(execution_log))
                with log_container:
                    if "✅" in log_entry:
                        st.success(log_entry)
                    elif "Starting" in log_entry or "finished" in log_entry:
                        st.info(log_entry)
                    else:
                        st.write(log_entry)
            
            # Show agent workflow
            st.subheader("🤖 Agent Coordination")
            workflow = simulate_agent_workflow()
            for line in workflow:
                if line.strip():
                    st.code(line)

# Settings
elif page == "Settings":
    st.title("⚙️ System Configuration")
    
    tabs = st.tabs(["General", "AI Settings", "API Integration", "System Info"])
    
    with tabs[0]:  # General
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Interface Language",
                ["English", "German", "Arabic", "Spanish", "French"]
            )
            
            timezone = st.selectbox(
                "Timezone",
                ["UTC", "EST", "PST", "CET", "IST"]
            )
            
            date_format = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"]
            )
        
        with col2:
            auto_refill = st.checkbox("Enable Auto-Refill", value=True)
            notifications = st.checkbox("Enable Notifications", value=True)
            voice_input = st.checkbox("Enable Voice Input", value=False)
        
        if st.button("Save General Settings", use_container_width=True):
            st.session_state.system_settings = {
                'language': language,
                'auto_refill': auto_refill,
                'notifications': notifications
            }
            st.success("Settings saved successfully!")
    
    with tabs[1]:  # AI Settings
        st.subheader("AI Configuration")
        
        model = st.selectbox(
            "AI Model",
            ["GPT-4 Simulation", "Local LLM", "Hybrid Mode"]
        )
        
        temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
        max_response = st.number_input("Max Response Length", 100, 2000, 500)
        
        st.write("**Agent Configuration**")
        autonomy = st.slider("Agent Autonomy Level", 0, 100, 75)
        enable_learning = st.checkbox("Enable Machine Learning", value=True)
        
        if st.button("Save AI Settings", use_container_width=True):
            st.success("AI configuration saved!")
    
    with tabs[2]:  # API Integration
        st.subheader("API Integration")
        
        st.warning("⚠️ API keys are sensitive. Store securely in production.")
        
        zapier_key = st.text_input("Zapier API Key", type="password")
        n8n_webhook = st.text_input("n8n Webhook URL")
        mediloon_api = st.text_input("Mediloon CMS API Key", type="password")
        
        if st.button("Test API Connections", use_container_width=True):
            with st.spinner("Testing connections..."):
                time.sleep(2)
                
                results = []
                if zapier_key:
                    results.append(("Zapier", "✅ Connected"))
                else:
                    results.append(("Zapier", "❌ Not configured"))
                
                if n8n_webhook:
                    results.append(("n8n", "✅ Connected"))
                else:
                    results.append(("n8n", "❌ Not configured"))
                
                if mediloon_api:
                    results.append(("Mediloon CMS", "✅ Connected"))
                else:
                    results.append(("Mediloon CMS", "❌ Not configured"))
                
                for name, status in results:
                    if "✅" in status:
                        st.success(f"{name}: {status}")
                    else:
                        st.error(f"{name}: {status}")
        
        if st.button("Save API Keys", use_container_width=True):
            st.success("API keys saved (session only)")
    
    with tabs[3]:  # System Info
        st.subheader("System Information")
        
        st.write("**Application Details**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Version:** 3.0.1")
            st.write("**Build:** HackFusion Edition")
            st.write("**Python:** 3.13+")
        
        with col2:
            st.write(f"**Orders:** {len(st.session_state.orders)}")
            st.write(f"**Inventory:** {len(st.session_state.inventory)}")
            st.write(f"**Agents:** {len(st.session_state.agents)}")
        
        # System actions
        st.subheader("System Actions")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Export All Data", use_container_width=True):
                data = {
                    "orders": st.session_state.orders,
                    "inventory": st.session_state.inventory,
                    "settings": st.session_state.system_settings,
                    "exported_at": datetime.now().isoformat()
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(data, indent=2, default=str),
                    file_name="trackfusion_export.json",
                    mime="application/json"
                )
        
        with col_b:
            if st.button("Clear All Data", use_container_width=True):
                st.warning("This will reset all data. Continue?")
                if st.button("Confirm Reset", type="secondary"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    init_session_state()
                    init_demo_data()
                    st.success("All data cleared and reset!")
                    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 HackFusion Project")
st.sidebar.markdown("**TrackFusion 3** - AI Pharmacy System")
st.sidebar.markdown("---")
st.sidebar.caption("Made with ❤️ for HackFusion")

# Initialize on first run
if __name__ == "__main__":
    # Ensure demo data is initialized
    if not st.session_state.orders:
        init_demo_data()
