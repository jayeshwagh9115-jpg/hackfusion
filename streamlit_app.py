import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import time

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
    """Predict when medicines will run out"""
    predictions = {}
    
    for medicine, info in MEDICINES.items():
        # Get recent orders for this medicine
        recent_orders = [o for o in st.session_state.orders 
                        if o["medicine"] == medicine]
        
        if recent_orders:
            # Calculate consumption
            total_qty = sum([o["quantity"] for o in recent_orders])
            days_span = max([(datetime.now() - o["date"]).days for o in recent_orders]) or 30
            daily_consumption = total_qty / days_span
            
            # Find inventory
            inv_item = next((i for i in st.session_state.inventory 
                           if i["medicine"] == medicine), None)
            
            if inv_item and daily_consumption > 0:
                days_until_empty = inv_item["current_stock"] / daily_consumption
                refill_date = datetime.now() + timedelta(days=days_until_empty)
                
                predictions[medicine] = {
                    "refill_date": refill_date,
                    "days_until_empty": int(days_until_empty),
                    "daily_consumption": round(daily_consumption, 2),
                    "current_stock": inv_item["current_stock"],
                    "reorder_level": inv_item["reorder_level"],
                    "urgency": "CRITICAL" if days_until_empty < 3 else 
                              "HIGH" if days_until_empty < 7 else 
                              "MEDIUM" if days_until_empty < 14 else "LOW"
                }
    
    st.session_state.predictions = predictions
    return predictions

def process_natural_language(input_text):
    """Process natural language for medicine ordering"""
    input_text = input_text.lower()
    detected_meds = []
    
    # Simple keyword matching
    for med in MEDICINES.keys():
        if med.lower() in input_text:
            detected_meds.append(med)
    
    # Look for quantities
    quantities = []
    words = input_text.split()
    quantity_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    
    for word in words:
        if word.isdigit():
            quantities.append(int(word))
        elif word in quantity_map:
            quantities.append(quantity_map[word])
    
    if not quantities:
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
        critical = sum(1 for p in predictions.values() if p["urgency"] in ["CRITICAL", "HIGH"])
        st.metric("Urgent Refills", critical)
    with col4:
        st.metric("Active Agents", sum(1 for a in st.session_state.agents.values() if a["status"] == "Active"))
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    
    tab1, tab2 = st.tabs(["Recent Orders", "Inventory Status"])
    
    with tab1:
        if st.session_state.orders:
            recent_orders = sorted(st.session_state.orders, key=lambda x: x["date"], reverse=True)[:10]
            orders_df = pd.DataFrame(recent_orders)
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("No orders yet")
    
    with tab2:
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            st.dataframe(inv_df, use_container_width=True)
            
            # Simple chart
            chart_data = inv_df[["medicine", "current_stock"]].set_index("medicine")
            st.bar_chart(chart_data)
    
    # Alerts
    st.subheader("⚠️ System Alerts")
    
    alerts = []
    for item in st.session_state.inventory:
        if item["current_stock"] < item["reorder_level"]:
            alerts.append(f"Low stock: {item['medicine']} ({item['current_stock']} left)")
    
    if alerts:
        for alert in alerts[:3]:
            st.warning(alert)
    else:
        st.success("✅ All systems operational")

# AI Assistant
elif page == "AI Assistant":
    st.title("🎤 AI Ordering Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Chat history
        for chat in st.session_state.chat_history[-5:]:
            if chat["role"] == "user":
                st.chat_message("user").write(chat["content"])
            else:
                st.chat_message("assistant").write(chat["content"])
        
        # Chat input
        user_input = st.chat_input("Type your medicine order...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("Processing..."):
                medicines, quantities = process_natural_language(user_input)
                
                if medicines:
                    add_to_cart(medicines, quantities)
                    response = f"Added {', '.join(medicines)} to your cart."
                    
                    # Check prescriptions
                    presc_meds = [m for m in medicines if MEDICINES[m]["prescription_required"]]
                    if presc_meds:
                        response += f"\n\n⚠️ Note: {', '.join(presc_meds)} require prescription."
                else:
                    response = "I can help you order medicines. Try saying 'I need Metformin' or 'Order Ibuprofen'."
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
    
    with col2:
        st.subheader("🛒 Your Cart")
        
        if st.session_state.cart:
            for item in st.session_state.cart:
                with st.expander(f"{item['medicine']} ({item['quantity']}x)"):
                    st.write(f"Price: ${item['price']:.2f}")
                    st.write(f"Total: ${item['price'] * item['quantity']:.2f}")
                    st.write(f"Category: {item['category']}")
            
            summary = generate_order_summary()
            if summary:
                st.markdown("---")
                st.write(f"**Items:** {summary['total_items']}")
                st.write(f"**Subtotal:** ${summary['subtotal']:.2f}")
                st.write(f"**Tax:** ${summary['tax']:.2f}")
                st.write(f"**Total:** ${summary['subtotal'] + summary['tax'] + summary['shipping']:.2f}")
                
                if st.button("Checkout", type="primary"):
                    workflow = simulate_agent_workflow()
                    st.success("Order placed!")
                    for line in workflow:
                        st.code(line)
                    
                    # Clear cart
                    st.session_state.cart = []
                    st.rerun()
            
            if st.button("Clear Cart"):
                st.session_state.cart = []
                st.rerun()
        else:
            st.info("Cart is empty")

# Predictive Engine
elif page == "Predictive Engine":
    st.title("🔮 Predictive Ordering Engine")
    
    predictions = predict_refill_dates()
    
    if predictions:
        # Summary
        critical = sum(1 for p in predictions.values() if p["urgency"] == "CRITICAL")
        high = sum(1 for p in predictions.values() if p["urgency"] == "HIGH")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Critical Refills", critical)
        with col2:
            st.metric("High Priority", high)
        
        # Predictions table
        st.subheader("📊 Refill Predictions")
        
        pred_data = []
        for med, pred in predictions.items():
            pred_data.append({
                "Medicine": med,
                "Current Stock": pred["current_stock"],
                "Days Until Empty": pred["days_until_empty"],
                "Refill Date": pred["refill_date"].strftime("%Y-%m-%d"),
                "Urgency": pred["urgency"]
            })
        
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True)
        
        # Auto-refill
        needs_refill = [med for med, pred in predictions.items() 
                       if pred["urgency"] in ["CRITICAL", "HIGH"]]
        
        if needs_refill:
            st.warning(f"⚠️ {len(needs_refill)} medicines need urgent refill")
            
            if st.button("Auto-Order Critical Items"):
                quantities = [3] * len(needs_refill)
                add_to_cart(needs_refill, quantities)
                st.success(f"Added {len(needs_refill)} items to cart!")
        else:
            st.success("✅ All inventory levels are good")
    
    else:
        st.info("No prediction data available")

# Agent System
elif page == "Agent System":
    st.title("🤖 Multi-Agent System")
    
    st.markdown("""
    ### Agent Architecture
    
    TrackFusion 3 uses a multi-agent system to automate pharmacy operations:
    """)
    
    # Agent status
    col1, col2, col3, col4 = st.columns(4)
    
    agents = [
        ("Ordering Agent", "🛒", "Handles customer orders"),
        ("Safety Agent", "🛡️", "Checks prescriptions"),
        ("Forecast Agent", "📈", "Predicts refills"),
        ("Procurement Agent", "📦", "Manages suppliers")
    ]
    
    for idx, (name, icon, desc) in enumerate(agents):
        with [col1, col2, col3, col4][idx]:
            status = st.session_state.agents[name]["status"]
            color = "green" if status == "Active" else "orange"
            
            st.markdown(f"""
            <div class="agent-card">
                <h3>{icon} {name}</h3>
                <p><strong>Status:</strong> <span style="color:{color}">{status}</span></p>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Workflow simulation
    st.subheader("⚙️ Workflow Simulation")
    
    if st.button("Run Complete Workflow", type="primary"):
        with st.spinner("Executing multi-agent workflow..."):
            workflow = simulate_agent_workflow()
            
            for line in workflow:
                if line.startswith("🤖"):
                    st.success(line)
                elif line.startswith("🛒") or line.startswith("🛡️") or line.startswith("📈") or line.startswith("📦"):
                    st.info(line)
                elif line.startswith("🔗"):
                    st.warning(line)
                elif line.strip():
                    st.code(line)
            
            st.success("✅ Workflow completed successfully!")

# Inventory Management
elif page == "Inventory":
    st.title("📦 Inventory Management")
    
    # Summary
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
        
        # Add status column
        inv_df["status"] = inv_df.apply(
            lambda row: "⚠️ Low" if row["current_stock"] < row["reorder_level"] else "✅ Good",
            axis=1
        )
        
        # Editable dataframe
        edited_df = st.data_editor(
            inv_df[["medicine", "current_stock", "reorder_level", "status", "supplier"]],
            column_config={
                "current_stock": st.column_config.NumberColumn(min_value=0),
                "reorder_level": st.column_config.NumberColumn(min_value=1)
            },
            use_container_width=True,
            num_rows="dynamic"
        )
        
        if st.button("Update Inventory"):
            for idx, row in edited_df.iterrows():
                if idx < len(st.session_state.inventory):
                    st.session_state.inventory[idx]["current_stock"] = row["current_stock"]
                    st.session_state.inventory[idx]["reorder_level"] = row["reorder_level"]
            st.success("Inventory updated!")
    
    # Reorder suggestions
    st.subheader("📋 Reorder Suggestions")
    
    suggestions = []
    for item in st.session_state.inventory:
        if item["current_stock"] < item["reorder_level"]:
            suggestions.append({
                "Medicine": item["medicine"],
                "Current": item["current_stock"],
                "Reorder At": item["reorder_level"],
                "Supplier": item["supplier"],
                "Order Qty": max(30, item["reorder_level"] * 2)
            })
    
    if suggestions:
        sugg_df = pd.DataFrame(suggestions)
        st.dataframe(sugg_df, use_container_width=True)
        
        meds_to_order = [s["Medicine"] for s in suggestions]
        quantities = [s["Order Qty"] for s in suggestions]
        
        if st.button("Generate Purchase Orders"):
            add_to_cart(meds_to_order, quantities)
            st.success(f"Added {len(meds_to_order)} items to cart for reorder!")
    else:
        st.success("✅ All stock levels are adequate")

# Workflow Automation
elif page == "Workflow":
    st.title("⚙️ MCP & Workflow Automation")
    
    st.markdown("""
    ### Multiple Connection Platform Integration
    
    TrackFusion 3 connects with external tools for end-to-end automation:
    """)
    
    # MCP Status
    mcp_services = {
        "Zapier": ("✅ Connected", "Email/SMS notifications"),
        "n8n": ("✅ Connected", "API automation"),
        "Mediloon CMS": ("✅ Connected", "Inventory sync"),
        "Supplier APIs": ("⚠️ Partial", "Order automation")
    }
    
    cols = st.columns(4)
    for idx, (service, (status, desc)) in enumerate(mcp_services.items()):
        with cols[idx]:
            st.metric(service, status.split()[0], desc)
    
    # Workflow designer
    st.subheader("🎨 Workflow Designer")
    
    workflow_type = st.selectbox(
        "Select Workflow",
        ["Order Fulfillment", "Inventory Replenishment", "Customer Notification", "Emergency Refill"]
    )
    
    st.write(f"**Selected:** {workflow_type}")
    st.write("**Steps:**")
    
    steps = {
        "Order Fulfillment": [
            "1. Customer places order",
            "2. AI Assistant processes request",
            "3. Safety Agent validates prescription",
            "4. Inventory system checks stock",
            "5. Payment processed",
            "6. Supplier notified via API",
            "7. Order shipped",
            "8. Customer notified"
        ],
        "Inventory Replenishment": [
            "1. Low stock detected",
            "2. Forecast Agent analyzes needs",
            "3. Procurement Agent generates PO",
            "4. Supplier order placed",
            "5. Inventory updated on arrival",
            "6. System notified of restock"
        ]
    }
    
    for step in steps.get(workflow_type, ["Workflow steps will appear here"]):
        st.write(step)
    
    if st.button("Execute Workflow", type="primary"):
        with st.spinner("Running workflow..."):
            time.sleep(2)
            workflow = simulate_agent_workflow()
            st.success("Workflow executed!")
            
            with st.expander("View Execution Log"):
                for line in workflow:
                    st.code(line)

# Settings
elif page == "Settings":
    st.title("⚙️ System Configuration")
    
    tab1, tab2, tab3 = st.tabs(["General", "AI Settings", "System Info"])
    
    with tab1:
        st.subheader("General Settings")
        
        # Language
        language = st.selectbox(
            "Interface Language",
            ["English", "German", "Arabic", "Spanish", "French"],
            index=0
        )
        
        # Features
        col1, col2 = st.columns(2)
        with col1:
            auto_refill = st.checkbox("Enable Auto-Refill", value=True)
            voice_input = st.checkbox("Enable Voice Input", value=False)
        with col2:
            notifications = st.checkbox("Enable Notifications", value=True)
            dark_mode = st.checkbox("Dark Mode", value=False)
        
        # Save button
        if st.button("Save Settings"):
            st.session_state.system_settings = {
                'language': language,
                'auto_refill': auto_refill,
                'notifications': notifications
            }
            st.success("Settings saved!")
    
    with tab2:
        st.subheader("AI Configuration")
        
        # Model settings
        model = st.selectbox(
            "AI Model",
            ["GPT-4 Simulation", "Local LLM", "Hybrid Mode"]
        )
        
        temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Response Length", 100, 2000, 500)
        
        # Agent settings
        st.write("**Agent Configuration**")
        autonomy = st.slider("Agent Autonomy", 0, 100, 75)
        learning = st.checkbox("Enable Machine Learning", value=True)
        
        if st.button("Save AI Config"):
            st.success("AI configuration saved!")
    
    with tab3:
        st.subheader("System Information")
        
        st.write("**Version:** TrackFusion 3.0")
        st.write("**Build:** HackFusion Edition")
        st.write(f"**Data:** {len(st.session_state.orders)} orders, {len(st.session_state.inventory)} items")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # System actions
        st.subheader("System Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Data"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                init_session_state()
                init_demo_data()
                st.success("All data cleared and reset!")
                st.rerun()
        
        with col2:
            if st.button("Export Data"):
                data = {
                    "orders": st.session_state.orders,
                    "inventory": st.session_state.inventory,
                    "settings": st.session_state.system_settings
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(data, indent=2, default=str),
                    file_name="trackfusion_data.json",
                    mime="application/json"
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 HackFusion Project")
st.sidebar.markdown("**TrackFusion 3** - AI Pharmacy System")
st.sidebar.markdown("[Report Issue](https://github.com)")
st.sidebar.markdown("---")

# Initialize on first run
if __name__ == "__main__":
    # Ensure data is initialized
    if not st.session_state.orders:
        init_demo_data()
