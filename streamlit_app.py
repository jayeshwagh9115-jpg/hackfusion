import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import time
import math
from typing import Dict, List, Tuple, Optional
import re

# Try to import optional menu
try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except ImportError:
    OPTION_MENU_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="TrackFusion 3 - AI Pharmacy System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONFIGURATION ====================

# Multi-language support
LANGUAGES = {
    "English": {
        "dashboard": "Dashboard",
        "ai_assistant": "AI Assistant",
        "predictive_engine": "Predictive Engine",
        "agent_system": "Agent System",
        "inventory": "Inventory",
        "workflow": "Workflow",
        "settings": "Settings",
        "welcome": "Welcome to TrackFusion 3",
        "subtitle": "AI-Driven Autonomous Pharmacy System"
    },
    "German": {
        "dashboard": "Dashboard",
        "ai_assistant": "KI-Assistent",
        "predictive_engine": "Vorhersage-Engine",
        "agent_system": "Agenten-System",
        "inventory": "Inventar",
        "workflow": "Workflow",
        "settings": "Einstellungen",
        "welcome": "Willkommen bei TrackFusion 3",
        "subtitle": "KI-gesteuertes autonomes Apothekensystem"
    },
    "Arabic": {
        "dashboard": "لوحة التحكم",
        "ai_assistant": "المساعد الذكي",
        "predictive_engine": "محرك التنبؤ",
        "agent_system": "نظام الوكلاء",
        "inventory": "المخزون",
        "workflow": "سير العمل",
        "settings": "الإعدادات",
        "welcome": "مرحبًا بكم في TrackFusion 3",
        "subtitle": "نظام الصيدلية الذاتية المدعوم بالذكاء الاصطناعي"
    }
}

# Medicine database with alternatives and variants
MEDICINES = {
    "Metformin": {
        "type": "Diabetes",
        "refill_days": 30,
        "prescription_required": True,
        "price": 25.50,
        "alternatives": ["Glimepiride", "Pioglitazone", "Sitagliptin"],
        "variants": ["500mg", "850mg", "1000mg"],
        "interactions": ["Insulin", "Sulfonylureas"],
        "common_names": ["metformin", "glucophage", "fortamet"]
    },
    "Lisinopril": {
        "type": "Blood Pressure",
        "refill_days": 30,
        "prescription_required": True,
        "price": 18.75,
        "alternatives": ["Enalapril", "Ramipril", "Losartan"],
        "variants": ["5mg", "10mg", "20mg", "40mg"],
        "interactions": ["Diuretics", "NSAIDs"],
        "common_names": ["lisinopril", "zestril", "prinivil"]
    },
    "Atorvastatin": {
        "type": "Cholesterol",
        "refill_days": 30,
        "prescription_required": True,
        "price": 32.00,
        "alternatives": ["Simvastatin", "Rosuvastatin", "Pravastatin"],
        "variants": ["10mg", "20mg", "40mg", "80mg"],
        "interactions": ["Grapefruit juice", "Cyclosporine"],
        "common_names": ["atorvastatin", "lipitor"]
    },
    "Ibuprofen": {
        "type": "Pain Relief",
        "refill_days": 0,
        "prescription_required": False,
        "price": 8.99,
        "alternatives": ["Acetaminophen", "Naproxen", "Aspirin"],
        "variants": ["200mg", "400mg", "600mg"],
        "interactions": ["Blood thinners", "SSRIs"],
        "common_names": ["ibuprofen", "advil", "motrin", "nurofen"]
    },
    "Vitamin D": {
        "type": "Supplement",
        "refill_days": 60,
        "prescription_required": False,
        "price": 12.50,
        "alternatives": ["Calcium + Vitamin D", "Multivitamin"],
        "variants": ["1000IU", "2000IU", "5000IU"],
        "interactions": ["Calcium supplements"],
        "common_names": ["vitamin d", "cholecalciferol", "calciferol"]
    },
    "Albuterol": {
        "type": "Asthma",
        "refill_days": 30,
        "prescription_required": True,
        "price": 28.50,
        "alternatives": ["Levalbuterol", "Salmeterol", "Formoterol"],
        "variants": ["Inhaler 90mcg", "Nebulizer solution"],
        "interactions": ["Beta-blockers", "Diuretics"],
        "common_names": ["albuterol", "ventolin", "proair", "proventil"]
    },
    "Levothyroxine": {
        "type": "Thyroid",
        "refill_days": 90,
        "prescription_required": True,
        "price": 15.25,
        "alternatives": ["Liothyronine", "Natural thyroid extract"],
        "variants": ["25mcg", "50mcg", "75mcg", "100mcg", "125mcg", "150mcg"],
        "interactions": ["Calcium", "Iron", "Antacids"],
        "common_names": ["levothyroxine", "synthroid", "levoxyl", "tirosint"]
    }
}

# Drug interaction database
DRUG_INTERACTIONS = {
    ("Metformin", "Insulin"): "Increased risk of hypoglycemia",
    ("Lisinopril", "NSAIDs"): "Reduced antihypertensive effect",
    ("Atorvastatin", "Grapefruit juice"): "Increased risk of side effects",
    ("Ibuprofen", "Blood thinners"): "Increased bleeding risk",
    ("Levothyroxine", "Calcium"): "Reduced absorption, take 4 hours apart"
}

# ==================== SESSION STATE ====================

def init_session_state():
    """Initialize all session state variables"""
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
            'notifications': True,
            'voice_enabled': False,
            'dark_mode': False
        },
        'conversation_context': {
            'awaiting_prescription': False,
            'awaiting_variant': False,
            'awaiting_confirmation': False,
            'last_medicine': None,
            'last_quantity': None
        },
        'notifications': [],
        'onboarding_complete': False,
        'current_conversation_id': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

# ==================== CORE FUNCTIONS ====================

def init_demo_data():
    """Initialize comprehensive demo data"""
    if not st.session_state.orders:
        for med_name, info in MEDICINES.items():
            for _ in range(random.randint(2, 6)):
                order_date = datetime.now() - timedelta(days=random.randint(0, 180))
                st.session_state.orders.append({
                    "id": f"ORD{random.randint(10000, 99999)}",
                    "medicine": med_name,
                    "variant": random.choice(info["variants"]),
                    "quantity": random.randint(1, 3),
                    "date": order_date,
                    "user_id": "demo_user",
                    "status": random.choice(["Delivered", "Processing", "Shipped", "Pending"]),
                    "prescription_verified": info["prescription_required"] and random.choice([True, False]),
                    "total_price": info["price"] * random.randint(1, 3),
                    "delivery_date": order_date + timedelta(days=random.randint(2, 7))
                })
    
    if not st.session_state.inventory:
        for med_name, info in MEDICINES.items():
            current_stock = random.randint(5, 150)
            st.session_state.inventory.append({
                "medicine": med_name,
                "category": info["type"],
                "current_stock": current_stock,
                "reorder_level": random.randint(10, 30),
                "last_ordered": datetime.now() - timedelta(days=random.randint(1, 60)),
                "supplier": random.choice(["PharmaCorp", "MediSupply", "HealthPlus", "Global Pharma", "BioMed Inc."]),
                "lead_time_days": random.randint(2, 7),
                "unit_price": info["price"],
                "total_value": current_stock * info["price"],
                "variants_available": info["variants"]
            })

def advanced_predict_refill_dates():
    """Advanced prediction engine with multiple algorithms"""
    predictions = {}
    
    for medicine, info in MEDICINES.items():
        # Get recent orders (last 180 days)
        recent_orders = [o for o in st.session_state.orders 
                        if o["medicine"] == medicine and 
                        (datetime.now() - o["date"]).days <= 180]
        
        if recent_orders and len(recent_orders) >= 2:
            # Algorithm 1: Moving Average
            quantities = [o["quantity"] for o in recent_orders]
            dates = [o["date"] for o in recent_orders]
            
            # Calculate daily consumption using weighted moving average
            sorted_orders = sorted(zip(dates, quantities), key=lambda x: x[0])
            total_days = (sorted_orders[-1][0] - sorted_orders[0][0]).days or 30
            
            # Give more weight to recent orders
            weights = np.linspace(0.5, 1.5, len(quantities))
            weighted_quantities = quantities * weights
            total_weighted_qty = sum(weighted_quantities)
            
            daily_consumption = total_weighted_qty / max(total_days, 7)
            
            # Find inventory
            inv_item = next((i for i in st.session_state.inventory 
                           if i["medicine"] == medicine), None)
            
            if inv_item and daily_consumption > 0:
                days_until_empty = inv_item["current_stock"] / daily_consumption
                
                # Algorithm 2: Seasonal adjustment (simulated)
                day_of_week_factor = 1.0
                current_day = datetime.now().weekday()
                if current_day in [0, 1]:  # Monday, Tuesday
                    day_of_week_factor = 1.1  # 10% higher consumption
                
                # Algorithm 3: Trend detection
                if len(quantities) >= 4:
                    first_half = sum(quantities[:len(quantities)//2])
                    second_half = sum(quantities[len(quantities)//2:])
                    trend_factor = second_half / first_half if first_half > 0 else 1.0
                    trend_factor = min(max(trend_factor, 0.7), 1.3)  # Cap between 0.7-1.3
                else:
                    trend_factor = 1.0
                
                # Apply adjustments
                adjusted_daily_consumption = daily_consumption * day_of_week_factor * trend_factor
                adjusted_days_until_empty = inv_item["current_stock"] / adjusted_daily_consumption
                
                refill_date = datetime.now() + timedelta(days=adjusted_days_until_empty)
                
                # Calculate confidence score
                data_points = len(recent_orders)
                date_range = total_days
                consistency = 1 - (np.std(quantities) / np.mean(quantities)) if np.mean(quantities) > 0 else 0.5
                
                confidence = min(95, 
                               60 +  # Base
                               min(data_points * 3, 15) +  # Data points bonus
                               min(date_range / 30 * 5, 10) +  # Time range bonus
                               consistency * 10)  # Consistency bonus
                
                urgency_level = "CRITICAL" if adjusted_days_until_empty < 3 else \
                               "HIGH" if adjusted_days_until_empty < 7 else \
                               "MEDIUM" if adjusted_days_until_empty < 14 else "LOW"
                
                predictions[medicine] = {
                    "refill_date": refill_date,
                    "days_until_empty": int(adjusted_days_until_empty),
                    "daily_consumption": round(adjusted_daily_consumption, 3),
                    "current_stock": inv_item["current_stock"],
                    "reorder_level": inv_item["reorder_level"],
                    "confidence": round(confidence),
                    "urgency": urgency_level,
                    "algorithm_used": "Weighted Moving Average with Adjustments",
                    "trend": "Increasing" if trend_factor > 1.05 else "Decreasing" if trend_factor < 0.95 else "Stable",
                    "next_order_qty": max(inv_item["reorder_level"] * 2, 
                                         int(adjusted_daily_consumption * 30 * 1.2))  # 20% buffer
                }
    
    st.session_state.predictions = predictions
    return predictions

def check_drug_interactions(medicines: List[str]) -> List[Dict]:
    """Check for potential drug interactions"""
    interactions = []
    
    for i in range(len(medicines)):
        for j in range(i + 1, len(medicines)):
            med1, med2 = medicines[i], medicines[j]
            
            # Check both orders
            if (med1, med2) in DRUG_INTERACTIONS:
                interactions.append({
                    "medicines": [med1, med2],
                    "interaction": DRUG_INTERACTIONS[(med1, med2)],
                    "severity": "Moderate"
                })
            elif (med2, med1) in DRUG_INTERACTIONS:
                interactions.append({
                    "medicines": [med1, med2],
                    "interaction": DRUG_INTERACTIONS[(med2, med1)],
                    "severity": "Moderate"
                })
    
    return interactions

def advanced_nlp_processing(input_text: str, context: Dict) -> Tuple[List[str], List[int], Dict]:
    """Advanced NLP processing with context awareness"""
    input_text = input_text.lower().strip()
    detected_meds = []
    quantities = []
    response_context = {}
    
    # Check for conversation context
    if context.get('awaiting_variant') and context.get('last_medicine'):
        # User is selecting a variant
        for variant in MEDICINES[context['last_medicine']]["variants"]:
            if variant.lower() in input_text:
                detected_meds = [context['last_medicine']]
                quantities = [context.get('last_quantity', 1)]
                response_context = {
                    'variant_selected': variant,
                    'awaiting_variant': False,
                    'awaiting_confirmation': True
                }
                return detected_meds, quantities, response_context
    
    if context.get('awaiting_confirmation') and context.get('last_medicine'):
        # User is confirming an order
        confirmation_words = ["yes", "yeah", "sure", "ok", "okay", "confirm", "proceed"]
        if any(word in input_text for word in confirmation_words):
            detected_meds = [context['last_medicine']]
            quantities = [context.get('last_quantity', 1)]
            response_context = {'awaiting_confirmation': False}
            return detected_meds, quantities, response_context
    
    # Standard medicine detection
    quantity_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "some": 2, "few": 3, "several": 3,
        "dozen": 12, "half dozen": 6
    }
    
    unit_map = {
        "boxes": 1, "box": 1, "packs": 1, "pack": 1,
        "bottles": 1, "bottle": 1, "tubes": 1, "tube": 1,
        "strips": 1, "strip": 1
    }
    
    # Extract quantities with units
    words = input_text.split()
    for i, word in enumerate(words):
        # Numeric quantities
        if word.isdigit():
            qty = int(word)
            # Check if next word is a unit
            if i + 1 < len(words) and words[i + 1] in unit_map:
                quantities.append(qty * unit_map[words[i + 1]])
            else:
                quantities.append(qty)
        # Word quantities
        elif word in quantity_map:
            qty = quantity_map[word]
            quantities.append(qty)
    
    # Detect medicines using common names and synonyms
    for med_name, info in MEDICINES.items():
        # Check against common names
        found = False
        for common_name in info["common_names"]:
            if common_name in input_text:
                detected_meds.append(med_name)
                found = True
                break
        # Check against full name if not already found
        if not found and med_name.lower() in input_text:
            detected_meds.append(med_name)
    
    # Remove duplicates while preserving order
    detected_meds = list(dict.fromkeys(detected_meds))
    
    # Handle special phrases
    if not detected_meds:
        if "pain" in input_text or "headache" in input_text:
            detected_meds = ["Ibuprofen"]
            response_context['suggestion'] = "For pain relief, I recommend Ibuprofen"
        elif "diabetes" in input_text or "blood sugar" in input_text:
            detected_meds = ["Metformin"]
            response_context['suggestion'] = "For diabetes management, I recommend Metformin"
        elif "blood pressure" in input_text or "hypertension" in input_text:
            detected_meds = ["Lisinopril"]
            response_context['suggestion'] = "For blood pressure, I recommend Lisinopril"
        elif "cholesterol" in input_text:
            detected_meds = ["Atorvastatin"]
            response_context['suggestion'] = "For cholesterol management, I recommend Atorvastatin"
    
    # Match quantities to medicines
    if detected_meds and not quantities:
        quantities = [1] * len(detected_meds)
    elif len(quantities) > len(detected_meds):
        quantities = quantities[:len(detected_meds)]
    elif len(quantities) < len(detected_meds):
        quantities = quantities + [1] * (len(detected_meds) - len(quantities))
    
    return detected_meds, quantities, response_context

def generate_ai_response(user_input: str, medicines: List[str], quantities: List[int], 
                         context: Dict) -> Tuple[str, Dict]:
    """Generate intelligent AI response with conversation management"""
    response = ""
    new_context = context.copy()
    
    if not medicines:
        # No medicines detected
        responses = [
            "I can help you order medicines. Try saying things like:\n"
            "'I need Metformin for my diabetes'\n"
            "'Order two boxes of Ibuprofen'\n"
            "'My blood pressure medication needs refill'",
            
            "Please tell me what medicines you need. I can help with:\n"
            "• Medicine ordering\n• Refill reminders\n• Prescription validation\n"
            "• Drug interaction checks\n• Alternative suggestions",
            
            "I understand you're looking for medicines. Could you please specify:\n"
            "• The medicine name\n• The quantity needed\n• Any specific requirements?"
        ]
        response = random.choice(responses)
    
    elif len(medicines) == 1:
        med = medicines[0]
        qty = quantities[0]
        info = MEDICINES[med]
        
        # Check if prescription is required
        if info["prescription_required"]:
            response = f"I found {med} in your order. "
            response += "This medicine requires a prescription. "
            
            if not context.get('awaiting_prescription'):
                response += f"Do you have a prescription for {med}? (yes/no)"
                new_context['awaiting_prescription'] = True
                new_context['last_medicine'] = med
                new_context['last_quantity'] = qty
            else:
                if "yes" in user_input.lower():
                    response = f"Great! Prescription verified for {med}. "
                    response += f"Would you like the {random.choice(info['variants'])} variant? "
                    new_context['awaiting_prescription'] = False
                    new_context['awaiting_variant'] = True
                else:
                    response = f"I understand. {med} requires a valid prescription. "
                    response += "Would you like me to suggest an alternative? "
                    new_context['awaiting_prescription'] = False
        
        else:
            # No prescription required
            response = f"I've added {qty} {med} to your cart. "
            
            if len(info["variants"]) > 1 and not context.get('awaiting_variant'):
                response += f"Which variant would you prefer? Available: {', '.join(info['variants'])}"
                new_context['awaiting_variant'] = True
                new_context['last_medicine'] = med
                new_context['last_quantity'] = qty
            elif context.get('awaiting_variant') and context.get('variant_selected'):
                variant = context['variant_selected']
                response = f"Perfect! I've added {qty} {med} ({variant}) to your cart. "
                response += "Would you like to confirm this order? (yes/no)"
                new_context['awaiting_confirmation'] = True
                new_context['awaiting_variant'] = False
            elif context.get('awaiting_confirmation'):
                response = f"Order confirmed! {qty} {med} has been added to your cart. "
                response += "Is there anything else you'd like to order?"
                new_context['awaiting_confirmation'] = False
            else:
                # Suggest alternatives
                if info["alternatives"]:
                    alt = random.choice(info["alternatives"])
                    response += f"\n\n💡 **Alternative suggestion:** Consider {alt} as an alternative to {med}. "
                
                response += "\n\nIs there anything else you'd like to order?"
    
    else:
        # Multiple medicines
        med_list = ", ".join(medicines)
        response = f"I've added {med_list} to your cart. "
        
        # Check for drug interactions
        interactions = check_drug_interactions(medicines)
        if interactions:
            response += "\n\n⚠️ **Drug Interaction Alert:**"
            for interaction in interactions[:2]:  # Show max 2 interactions
                meds_str = " and ".join(interaction["medicines"])
                response += f"\n• {meds_str}: {interaction['interaction']}"
        
        # Check prescription requirements
        presc_meds = [m for m in medicines if MEDICINES[m]["prescription_required"]]
        if presc_meds:
            response += f"\n\n📋 **Prescription Note:** {', '.join(presc_meds)} require prescription validation."
        
        response += "\n\nWould you like to proceed to checkout or add more items?"
    
    return response, new_context

def add_to_cart_with_details(medicine: str, quantity: int, variant: str = None, 
                            prescription_verified: bool = False):
    """Add item to cart with detailed information"""
    info = MEDICINES[medicine]
    
    cart_item = {
        "medicine": medicine,
        "quantity": quantity,
        "price": info["price"],
        "prescription_required": info["prescription_required"],
        "prescription_verified": prescription_verified,
        "category": info["type"],
        "variant": variant or random.choice(info["variants"]),
        "added_at": datetime.now(),
        "interactions": [],
        "total_price": info["price"] * quantity
    }
    
    # Check interactions with existing cart items
    for item in st.session_state.cart:
        interactions = check_drug_interactions([medicine, item["medicine"]])
        if interactions:
            cart_item["interactions"].extend(interactions)
    
    st.session_state.cart.append(cart_item)

def simulate_advanced_agent_workflow(order_type: str = "standard"):
    """Simulate advanced multi-agent workflow with MCP integration"""
    workflow = []
    
    # Agent initialization
    workflow.append("🤖 **Multi-Agent System Initialized**")
    workflow.append("")
    
    # Ordering Agent
    workflow.append("🛒 **Ordering Agent**")
    workflow.append("  ↳ Received customer order request")
    workflow.append("  ↳ Parsed natural language input")
    workflow.append("  ↳ Extracted medicine names and quantities")
    workflow.append("  ↳ Created order structure")
    workflow.append("  ↳ Forwarded to Safety Agent")
    workflow.append("")
    
    # Safety Agent
    workflow.append("🛡️ **Safety Agent**")
    workflow.append("  ↳ Checking prescription requirements...")
    workflow.append("  ↳ Validating drug interactions...")
    workflow.append("  ↳ Verifying dosage information...")
    workflow.append("  ↳ Cross-referencing with patient history...")
    workflow.append("  ↳ Safety clearance granted")
    workflow.append("")
    
    # Forecast Agent
    workflow.append("📈 **Forecast Agent**")
    workflow.append("  ↳ Analyzing consumption patterns...")
    workflow.append("  ↳ Calculating predictive models...")
    workflow.append("  ↳ Identifying seasonal trends...")
    workflow.append("  ↳ Generating refill predictions...")
    workflow.append("  ↳ Inventory forecast updated")
    workflow.append("")
    
    # Procurement Agent
    workflow.append("📦 **Procurement Agent**")
    workflow.append("  ↳ Generating purchase order...")
    workflow.append("  ↳ Analyzing supplier pricing...")
    workflow.append("  ↳ Negotiating optimal rates...")
    workflow.append("  ↳ Confirming delivery timelines...")
    workflow.append("  ↳ Purchase order finalized")
    workflow.append("")
    
    # MCP Integration
    workflow.append("🔗 **MCP Integration & Real-World Execution**")
    workflow.append("")
    workflow.append("  📧 **Zapier Automation:**")
    workflow.append("    ↳ Order confirmation email sent to customer")
    workflow.append("    ↳ SMS notification dispatched")
    workflow.append("    ↳ WhatsApp alert triggered")
    workflow.append("    ↳ Prescription reminder scheduled")
    workflow.append("")
    workflow.append("  ⚡ **n8n Workflows:**")
    workflow.append("    ↳ Supplier API called for stock check")
    workflow.append("    ↳ Distributor order placed automatically")
    workflow.append("    ↳ Payment gateway integrated")
    workflow.append("    ↳ Shipping label generated")
    workflow.append("")
    workflow.append("  🌐 **Webhooks & CMS Integration:**")
    workflow.append("    ↳ Mediloon CMS updated in real-time")
    workflow.append("    ↳ Inventory synchronized across systems")
    workflow.append("    ↳ Shopping cart cleared and reset")
    workflow.append("    ↳ Order tracking activated")
    workflow.append("")
    workflow.append("✅ **Workflow Completed Successfully**")
    workflow.append("")
    workflow.append("📊 **Performance Metrics:**")
    workflow.append("  ↳ Processing Time: 2.3 seconds")
    workflow.append("  ↳ Accuracy: 99.7%")
    workflow.append("  ↳ Cost Savings: 15.2%")
    workflow.append("  ↳ Customer Satisfaction: 4.8/5.0")
    
    return workflow

def generate_prescription_report(medicine: str, patient_info: Dict) -> Dict:
    """Generate prescription validation report"""
    info = MEDICINES[medicine]
    
    report = {
        "medicine": medicine,
        "patient_name": patient_info.get("name", "Not Provided"),
        "patient_age": patient_info.get("age", "Not Provided"),
        "validation_date": datetime.now().isoformat(),
        "requirements": {
            "prescription_required": info["prescription_required"],
            "valid_dosage": True,
            "no_contraindications": True,
            "appropriate_for_age": patient_info.get("age", 30) >= 18 if info["type"] != "Pediatric" else True
        },
        "safety_checks": {
            "drug_interactions": check_drug_interactions([medicine]),
            "allergy_check": "No known allergies",
            "renal_function": "Normal",
            "hepatic_function": "Normal"
        },
        "status": "APPROVED" if info["prescription_required"] else "NOT_REQUIRED",
        "prescriber_recommendation": f"Prescription validated for {medicine}. "
                                    f"Standard dosage: {random.choice(info['variants'])}"
    }
    
    return report

# ==================== ONBOARDING QUESTIONNAIRE ====================

def show_onboarding_questionnaire():
    """Show first-time user onboarding questionnaire"""
    st.subheader("👋 Welcome to TrackFusion 3!")
    st.markdown("Please help us personalize your experience by answering a few questions:")
    
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", placeholder="John Doe")
            age = st.number_input("Age*", min_value=1, max_value=120, value=30)
            phone = st.text_input("Phone Number", placeholder="+1 234 567 8900")
        
        with col2:
            email = st.text_input("Email Address*", placeholder="john@example.com")
            emergency_contact = st.text_input("Emergency Contact", placeholder="Jane Doe - +1 987 654 3210")
        
        st.markdown("### Medical Information (Optional but Recommended)")
        
        medical_conditions = st.multiselect(
            "Medical Conditions",
            ["Diabetes", "Hypertension", "Asthma", "Thyroid Disorder", 
             "High Cholesterol", "Heart Disease", "Arthritis", "Migraine",
             "Anxiety/Depression", "Other"]
        )
        
        allergies = st.text_area("Known Allergies", placeholder="Penicillin, Sulfa drugs, etc.")
        
        regular_medications = st.text_area(
            "Current Regular Medications", 
            placeholder="List any medications you're currently taking"
        )
        
        preferences = st.multiselect(
            "Communication Preferences",
            ["Email", "SMS", "WhatsApp", "In-App Notifications", "Phone Call"],
            default=["Email", "In-App Notifications"]
        )
        
        consent = st.checkbox(
            "I consent to using this information to personalize my pharmacy experience "
            "and receive medication reminders.*"
        )
        
        submitted = st.form_submit_button("Complete Onboarding", type="primary")
        
        if submitted:
            if not name or not email or not consent:
                st.error("Please fill in all required fields (*) and give consent.")
            else:
                st.session_state.user_profile = {
                    "name": name,
                    "age": age,
                    "email": email,
                    "phone": phone if phone else "Not provided",
                    "emergency_contact": emergency_contact if emergency_contact else "Not provided",
                    "medical_conditions": medical_conditions,
                    "allergies": allergies if allergies else "None reported",
                    "regular_medications": regular_medications if regular_medications else "None reported",
                    "preferences": preferences,
                    "onboarding_date": datetime.now().isoformat(),
                    "risk_level": "Low" if age < 65 and not medical_conditions else "Medium"
                }
                st.session_state.onboarding_complete = True
                st.success("🎉 Onboarding completed successfully! Your profile has been saved.")
                st.rerun()

# ==================== NOTIFICATION SYSTEM ====================

def add_notification(notification_type: str, message: str, priority: str = "medium"):
    """Add a notification to the system"""
    notification = {
        "id": f"NOT{random.randint(10000, 99999)}",
        "type": notification_type,
        "message": message,
        "priority": priority,
        "timestamp": datetime.now(),
        "read": False
    }
    st.session_state.notifications.append(notification)

def show_notifications():
    """Display notifications panel"""
    if st.session_state.notifications:
        unread = [n for n in st.session_state.notifications if not n["read"]]
        
        if unread:
            with st.expander(f"🔔 Notifications ({len(unread)} unread)", expanded=False):
                for notification in unread[-5:]:  # Show last 5 unread
                    icon = "🔴" if notification["priority"] == "high" else "🟡" if notification["priority"] == "medium" else "🔵"
                    st.markdown(f"{icon} **{notification['type']}**")
                    st.write(notification["message"])
                    st.caption(notification["timestamp"].strftime("%Y-%m-%d %H:%M"))
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("Mark as read", key=f"read_{notification['id']}"):
                            notification["read"] = True
                            st.rerun()
                    
                    st.markdown("---")

# ==================== VISUALIZATION ====================

def create_prediction_timeline(predictions: Dict):
    """Create a timeline visualization of predictions"""
    if not predictions:
        return
    
    timeline_data = []
    for med, pred in predictions.items():
        timeline_data.append({
            "Medicine": med,
            "Days Until Empty": pred["days_until_empty"],
            "Refill Date": pred["refill_date"],
            "Urgency": pred["urgency"]
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Sort by days until empty
    df = df.sort_values("Days Until Empty")
    
    # Create a simple timeline using emojis
    st.subheader("📅 Refill Timeline")
    
    for _, row in df.iterrows():
        days = row["Days Until Empty"]
        urgency = row["Urgency"]
        
        # Create visual timeline bar
        if days <= 7:
            timeline_bar = "🔴" + "█" * max(1, days) + " " * (14 - days) + f" {days} days"
        elif days <= 14:
            timeline_bar = "🟡" + "█" * days + " " * (14 - days) + f" {days} days"
        else:
            timeline_bar = "🟢" + "█" * min(14, days) + f" {days} days"
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write(f"**{row['Medicine']}**")
            st.write(timeline_bar)
        with col2:
            st.write(f"Refill: {row['Refill Date'].strftime('%Y-%m-%d')}")
            st.write(f"Status: {urgency}")

# ==================== MAIN APP LAYOUT ====================

# Initialize demo data
init_demo_data()

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.2rem;
        border-radius: 1rem;
        border-left: 5px solid #2563eb;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .agent-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 1rem;
        border: 2px solid #bae6fd;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .chat-user {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .chat-assistant {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%);
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        border-left: 4px solid #8b5cf6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .urgency-critical { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #dc2626; 
        font-weight: 800;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
    .urgency-high { 
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        color: #ea580c; 
        font-weight: 800;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
    .urgency-medium { 
        background: linear-gradient(135deg, #fef08a 0%, #fde047 100%);
        color: #ca8a04; 
        font-weight: 800;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
    .urgency-low { 
        background: linear-gradient(135deg, #bbf7d0 0%, #86efac 100%);
        color: #16a34a; 
        font-weight: 800;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        display: inline-block;
    }
    .prescription-badge {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }
    .language-selector {
        position: fixed;
        bottom: 1rem;
        right: 1rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced navigation
with st.sidebar:
    # Language selector
    current_lang = st.session_state.system_settings['language']
    lang_names = LANGUAGES[current_lang]
    
    st.markdown(f'<div class="main-header">💊 TrackFusion 3</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{lang_names["subtitle"]}</div>', unsafe_allow_html=True)
    
    # Language switch
    new_lang = st.selectbox("🌐 Language", list(LANGUAGES.keys()), 
                           index=list(LANGUAGES.keys()).index(current_lang),
                           key="language_selector")
    
    if new_lang != current_lang:
        st.session_state.system_settings['language'] = new_lang
        st.rerun()
    
    st.markdown("---")
    
    # Navigation
    if OPTION_MENU_AVAILABLE:
        selected = option_menu(
            menu_title=lang_names["dashboard"],
            options=[
                lang_names["dashboard"],
                lang_names["ai_assistant"],
                lang_names["predictive_engine"],
                lang_names["agent_system"],
                lang_names["inventory"],
                lang_names["workflow"],
                lang_names["settings"]
            ],
            icons=["speedometer", "robot", "graph-up-arrow", "people", 
                  "box-seam", "gear-wide-connected", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                "icon": {"color": "#2563eb", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#2563eb", "color": "white"},
            }
        )
        
        # Map back to English keys for processing
        page_map = {
            lang_names["dashboard"]: "Dashboard",
            lang_names["ai_assistant"]: "AI Assistant",
            lang_names["predictive_engine"]: "Predictive Engine",
            lang_names["agent_system"]: "Agent System",
            lang_names["inventory"]: "Inventory",
            lang_names["workflow"]: "Workflow",
            lang_names["settings"]: "Settings"
        }
        page = page_map[selected]
    else:
        page = st.selectbox(
            "Navigation",
            ["Dashboard", "AI Assistant", "Predictive Engine", "Agent System", 
             "Inventory", "Workflow", "Settings"]
        )
    
    # Show notifications in sidebar
    show_notifications()
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Orders", len(st.session_state.orders))
    with col2:
        st.metric("Cart", len(st.session_state.cart))
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Demo Data", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['system_settings', 'onboarding_complete']:
                del st.session_state[key]
        init_session_state()
        init_demo_data()
        st.success("Demo data reset!")
        st.rerun()
    
    if st.button("📋 View Prescriptions", use_container_width=True):
        st.session_state.page = "Prescriptions"
        st.rerun()

# ==================== PAGE: DASHBOARD ====================

if page == "Dashboard":
    # Check onboarding
    if not st.session_state.onboarding_complete:
        show_onboarding_questionnaire()
    else:
        st.title(f"🏥 {lang_names['welcome']}, {st.session_state.user_profile.get('name', 'User')}!")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", len(st.session_state.orders), 
                     delta=f"+{random.randint(1, 5)} today")
        with col2:
            total_value = sum(item.get("total_value", 0) for item in st.session_state.inventory)
            st.metric("Inventory Value", f"${total_value:,.2f}")
        with col3:
            predictions = advanced_predict_refill_dates()
            critical = sum(1 for p in predictions.values() if p.get("urgency") in ["CRITICAL", "HIGH"])
            st.metric("Urgent Refills", critical, 
                     delta="Action needed" if critical > 0 else "All good")
        with col4:
            active_agents = sum(1 for a in st.session_state.agents.values() if a["status"] == "Active")
            st.metric("Active Agents", f"{active_agents}/4")
        
        # Recent Activity with Tabs
        st.subheader("📈 Recent Activity & Insights")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Recent Orders", "Inventory Status", "Predictions", "Health Insights"])
        
        with tab1:
            if st.session_state.orders:
                recent_orders = sorted(st.session_state.orders, 
                                      key=lambda x: x["date"], reverse=True)[:8]
                
                for order in recent_orders:
                    with st.expander(f"📦 {order['medicine']} - {order['status']}"):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.write(f"**Order ID:** {order['id']}")
                            st.write(f"**Date:** {order['date'].strftime('%Y-%m-%d')}")
                        with col_b:
                            st.write(f"**Quantity:** {order['quantity']}")
                            st.write(f"**Price:** ${order.get('total_price', 0):.2f}")
                        with col_c:
                            if order.get('prescription_verified'):
                                st.success("✅ Prescription Verified")
                            elif MEDICINES.get(order['medicine'], {}).get('prescription_required'):
                                st.warning("⚠️ Prescription Required")
                        
                        if order.get('delivery_date'):
                            delivery_str = order['delivery_date'].strftime('%Y-%m-%d')
                            if order['delivery_date'] < datetime.now():
                                st.info(f"📬 Delivered on {delivery_str}")
                            else:
                                st.info(f"🚚 Estimated delivery: {delivery_str}")
            else:
                st.info("No orders yet. Start ordering through the AI Assistant!")
        
        with tab2:
            if st.session_state.inventory:
                # Low stock alert
                low_stock = [item for item in st.session_state.inventory 
                            if item["current_stock"] < item["reorder_level"]]
                
                if low_stock:
                    st.warning(f"⚠️ {len(low_stock)} items are low on stock!")
                    for item in low_stock[:3]:
                        st.error(f"{item['medicine']}: {item['current_stock']} left (Reorder at {item['reorder_level']})")
                
                # Inventory summary
                st.dataframe(
                    pd.DataFrame(st.session_state.inventory)[
                        ["medicine", "current_stock", "reorder_level", "supplier", "total_value"]
                    ].sort_values("current_stock"),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No inventory data available.")
        
        with tab3:
            if predictions:
                create_prediction_timeline(predictions)
                
                # Show prediction details
                st.subheader("📊 Prediction Details")
                
                pred_df = pd.DataFrame([
                    {
                        "Medicine": med,
                        "Days Left": pred["days_until_empty"],
                        "Confidence": f"{pred['confidence']}%",
                        "Trend": pred.get("trend", "Stable"),
                        "Next Order Qty": pred.get("next_order_qty", 0)
                    }
                    for med, pred in predictions.items()
                ])
                
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
            else:
                st.info("No prediction data available. Place some orders first!")
        
        with tab4:
            if st.session_state.user_profile:
                st.write("### 🩺 Health Profile Summary")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Name:** {st.session_state.user_profile.get('name', 'Not provided')}")
                    st.write(f"**Age:** {st.session_state.user_profile.get('age', 'Not provided')}")
                    st.write(f"**Risk Level:** {st.session_state.user_profile.get('risk_level', 'Not assessed')}")
                
                with col_b:
                    conditions = st.session_state.user_profile.get('medical_conditions', [])
                    if conditions:
                        st.write("**Conditions:**")
                        for condition in conditions:
                            st.write(f"• {condition}")
                    else:
                        st.write("**Conditions:** None reported")
                
                # Medication adherence score
                if st.session_state.orders:
                    recent_orders_count = len([o for o in st.session_state.orders 
                                             if (datetime.now() - o["date"]).days <= 30])
                    adherence_score = min(100, recent_orders_count * 15)
                    
                    st.progress(adherence_score / 100, 
                               text=f"Medication Adherence Score: {adherence_score}%")
                    
                    if adherence_score >= 80:
                        st.success("Excellent adherence! Keep it up!")
                    elif adherence_score >= 60:
                        st.info("Good adherence. Consider setting up auto-refill.")
                    else:
                        st.warning("Low adherence detected. Would you like refill reminders?")

# ==================== PAGE: AI ASSISTANT ====================

elif page == "AI Assistant":
    st.title("🎤 AI Ordering Assistant")
    
    # Voice/Text toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 Multi-Turn Conversation Interface")
    with col2:
        voice_enabled = st.checkbox("🎤 Voice Mode", 
                                   value=st.session_state.system_settings.get('voice_enabled', False))
        if voice_enabled != st.session_state.system_settings.get('voice_enabled', False):
            st.session_state.system_settings['voice_enabled'] = voice_enabled
            st.rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display conversation history
        for i, chat in enumerate(st.session_state.chat_history[-8:]):
            if chat["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <strong>You:</strong> {chat["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Parse and display AI response with formatting
                content = chat["content"]
                # Convert markdown-like formatting
                content = content.replace("**", "<strong>").replace("**", "</strong>")
                content = content.replace("\n", "<br>")
                
                st.markdown(f"""
                <div class="chat-assistant">
                    <strong>AI Assistant:</strong> {content}
                </div>
                """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    input_col1, input_col2 = st.columns([4, 1])
    
    with input_col1:
        user_input = st.text_area(
            "Type your message:",
            placeholder="Example: 'I need my monthly diabetes medication. I usually take Metformin 1000mg.'",
            height=100,
            key="chat_input"
        )
    
    with input_col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Process with advanced NLP
        with st.spinner("🤖 AI Assistant is thinking..."):
            medicines, quantities, nlp_context = advanced_nlp_processing(
                user_input, 
                st.session_state.conversation_context
            )
            
            # Generate AI response
            ai_response, new_context = generate_ai_response(
                user_input, medicines, quantities, 
                st.session_state.conversation_context
            )
            
            # Update conversation context
            st.session_state.conversation_context.update(new_context)
            
            # If medicines were detected and context allows, add to cart
            if medicines and not new_context.get('awaiting_prescription') \
               and not new_context.get('awaiting_variant') \
               and not new_context.get('awaiting_confirmation'):
                
                for med, qty in zip(medicines, quantities):
                    info = MEDICINES[med]
                    
                    # Check prescription requirement
                    prescription_verified = False
                    if info["prescription_required"]:
                        # In real app, this would check against a prescription database
                        prescription_verified = random.choice([True, False])
                        if not prescription_verified:
                            ai_response += f"\n\n⚠️ Prescription validation pending for {med}."
                    
                    add_to_cart_with_details(med, qty, prescription_verified=prescription_verified)
                
                # Add notification
                add_notification(
                    "Cart Updated",
                    f"Added {len(medicines)} item(s) to cart",
                    "low"
                )
            
            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Rerun to update UI
            st.rerun()
    
    # Cart side panel
    st.sidebar.markdown("### 🛒 Current Cart")
    
    if st.session_state.cart:
        total = 0
        prescription_items = []
        
        for i, item in enumerate(st.session_state.cart):
            with st.sidebar.expander(f"{item['medicine']} ({item['quantity']}x)"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Variant:** {item.get('variant', 'Standard')}")
                    st.write(f"**Price:** ${item['price']:.2f}")
                with col_b:
                    st.write(f"**Total:** ${item['total_price']:.2f}")
                    
                    if item['prescription_required']:
                        if item['prescription_verified']:
                            st.success("✅ Prescription verified")
                        else:
                            prescription_items.append(item['medicine'])
                            st.error("❌ Prescription required")
                
                # Show interactions if any
                if item.get('interactions'):
                    st.warning("⚠️ Potential drug interactions detected")
        
        # Calculate totals
        subtotal = sum(item['total_price'] for item in st.session_state.cart)
        tax = subtotal * 0.08
        shipping = 0 if subtotal > 50 else 5.99
        total = subtotal + tax + shipping
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Order Summary**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.sidebar.write(f"Items: {len(st.session_state.cart)}")
            st.sidebar.write(f"Subtotal: ${subtotal:.2f}")
        with col2:
            st.sidebar.write(f"Tax: ${tax:.2f}")
            st.sidebar.write(f"Shipping: ${shipping:.2f}")
        
        st.sidebar.markdown(f"### Total: ${total:.2f}")
        
        # Checkout button with validation
        if prescription_items:
            st.sidebar.error(f"⚠️ Prescription required for: {', '.join(prescription_items)}")
            checkout_disabled = True
        else:
            checkout_disabled = False
        
        if st.sidebar.button("✅ Proceed to Checkout", 
                           type="primary", 
                           disabled=checkout_disabled,
                           use_container_width=True):
            # Generate order
            order_id = f"ORD{random.randint(100000, 999999)}"
            
            # Add to orders
            for item in st.session_state.cart:
                st.session_state.orders.append({
                    "id": order_id,
                    "medicine": item["medicine"],
                    "variant": item.get("variant"),
                    "quantity": item["quantity"],
                    "date": datetime.now(),
                    "status": "Processing",
                    "prescription_verified": item["prescription_verified"],
                    "total_price": item["total_price"]
                })
            
            # Show success message
            st.sidebar.success(f"🎉 Order #{order_id} placed!")
            
            # Show agent workflow
            with st.sidebar.expander("🤖 View Agent Workflow"):
                workflow = simulate_advanced_agent_workflow()
                for line in workflow:
                    st.code(line)
            
            # Clear cart
            st.session_state.cart = []
            
            # Add notification
            add_notification(
                "Order Placed",
                f"Order #{order_id} has been placed successfully",
                "high"
            )
            
            st.rerun()
        
        if st.sidebar.button("🗑️ Clear Cart", use_container_width=True):
            st.session_state.cart = []
            st.rerun()
    
    else:
        st.sidebar.info("Your cart is empty. Start a conversation to add items!")

# ==================== PAGE: PREDICTIVE ENGINE ====================

elif page == "Predictive Engine":
    st.title("🔮 Advanced Predictive Ordering Engine")
    
    # Engine controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 365, 90)
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 70, 99, 80)
    with col3:
        algorithm = st.selectbox("Prediction Algorithm", 
                                ["Weighted Moving Average", 
                                 "Exponential Smoothing", 
                                 "Seasonal Decomposition"])
    
    # Generate predictions
    with st.spinner("🔄 Running advanced prediction algorithms..."):
        predictions = advanced_predict_refill_dates()
        time.sleep(0.5)  # Simulate processing
    
    if predictions:
        # Filter by confidence
        filtered_predictions = {k: v for k, v in predictions.items() 
                               if v.get("confidence", 0) >= confidence_threshold}
        
        # Summary dashboard
        st.subheader("📊 Prediction Dashboard")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            critical = sum(1 for p in filtered_predictions.values() 
                          if p.get("urgency") == "CRITICAL")
            st.metric("Critical", critical, delta="Immediate action" if critical > 0 else None)
        
        with metric_cols[1]:
            avg_confidence = np.mean([p.get("confidence", 0) 
                                     for p in filtered_predictions.values()])
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with metric_cols[2]:
            avg_days = np.mean([p.get("days_until_empty", 0) 
                               for p in filtered_predictions.values()])
            st.metric("Avg Days Left", f"{avg_days:.1f}")
        
        with metric_cols[3]:
            total_order_qty = sum(p.get("next_order_qty", 0) 
                                 for p in filtered_predictions.values())
            st.metric("Total Order Qty", total_order_qty)
        
        # Timeline visualization
        create_prediction_timeline(filtered_predictions)
        
        # Detailed predictions table
        st.subheader("📋 Detailed Predictions")
        
        pred_data = []
        for med, pred in filtered_predictions.items():
            urgency_class = f"urgency-{pred['urgency'].lower()}"
            pred_data.append({
                "Medicine": med,
                "Category": MEDICINES[med]["type"],
                "Current Stock": pred["current_stock"],
                "Days Until Empty": pred["days_until_empty"],
                "Refill Date": pred["refill_date"].strftime("%Y-%m-%d"),
                "Confidence": f"{pred['confidence']}%",
                "Urgency": f'<span class="{urgency_class}">{pred["urgency"]}</span>',
                "Next Order": pred.get("next_order_qty", 0),
                "Algorithm": pred.get("algorithm_used", "N/A")
            })
        
        # Convert to DataFrame for display
        pred_df = pd.DataFrame(pred_data)
        
        # Display with HTML for styling
        st.markdown(pred_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Auto-refill recommendations
        st.subheader("🤖 Smart Refill Recommendations")
        
        # Group by urgency
        critical_meds = [med for med, pred in filtered_predictions.items() 
                        if pred["urgency"] == "CRITICAL"]
        high_meds = [med for med, pred in filtered_predictions.items() 
                    if pred["urgency"] == "HIGH"]
        
        if critical_meds:
            st.error(f"🚨 **CRITICAL ALERT:** {len(critical_meds)} medicines need immediate refill!")
            
            for med in critical_meds:
                pred = filtered_predictions[med]
                with st.expander(f"🔴 {med} - CRITICAL (Only {pred['days_until_empty']} days left)"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Current Stock:** {pred['current_stock']}")
                        st.write(f"**Daily Consumption:** {pred['daily_consumption']}")
                        st.write(f"**Confidence:** {pred['confidence']}%")
                    with col_b:
                        st.write(f"**Recommended Qty:** {pred.get('next_order_qty', 0)}")
                        st.write(f"**Suggested Supplier:** {random.choice(['PharmaCorp', 'MediSupply', 'HealthPlus'])}")
                        st.write(f"**Lead Time:** {random.randint(2, 5)} days")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"🛒 Add to Cart", key=f"cart_{med}"):
                            add_to_cart_with_details(med, pred.get('next_order_qty', 1))
                            st.success(f"Added {med} to cart!")
                            st.rerun()
                    with col2:
                        if st.button(f"📧 Notify Supplier", key=f"notify_{med}"):
                            st.info(f"Supplier notification sent for {med}")
                    with col3:
                        if st.button(f"⏰ Schedule Refill", key=f"schedule_{med}"):
                            st.info(f"{med} scheduled for auto-refill on {pred['refill_date'].strftime('%Y-%m-%d')}")
        
        # Batch operations
        st.subheader("🔄 Batch Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Auto-Order All Critical", type="primary", use_container_width=True):
                for med in critical_meds:
                    pred = filtered_predictions[med]
                    add_to_cart_with_details(med, pred.get('next_order_qty', 1))
                st.success(f"Added {len(critical_meds)} critical items to cart!")
                st.rerun()
        
        with col2:
            if st.button("Generate Purchase Report", use_container_width=True):
                report_data = {
                    "generated_at": datetime.now().isoformat(),
                    "predictions": filtered_predictions,
                    "total_order_value": sum(
                        filtered_predictions[med].get('next_order_qty', 0) * MEDICINES[med]["price"]
                        for med in filtered_predictions.keys()
                    )
                }
                
                st.download_button(
                    label="📥 Download Report",
                    data=json.dumps(report_data, indent=2, default=str),
                    file_name="prediction_report.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("Schedule All Refills", use_container_width=True):
                for med, pred in filtered_predictions.items():
                    if pred["urgency"] in ["CRITICAL", "HIGH"]:
                        st.info(f"Scheduled {med} for {pred['refill_date'].strftime('%Y-%m-%d')}")
                st.success("All urgent refills scheduled!")
    
    else:
        st.info("No prediction data available. The system needs more order history to generate accurate predictions.")
        
        # Show onboarding prompt
        if not st.session_state.onboarding_complete:
            st.warning("Complete your onboarding questionnaire to get personalized predictions!")
            if st.button("Start Onboarding"):
                st.session_state.onboarding_complete = False
                st.rerun()

# ==================== PAGE: AGENT SYSTEM ====================

elif page == "Agent System":
    st.title("🤖 Multi-Agent Architecture")
    
    st.markdown("""
    ### LangChain/LangGraph-based Agent System
    
    TrackFusion 3 implements a sophisticated multi-agent architecture using simulated 
    LangChain/LangGraph patterns for autonomous pharmacy operations.
    """)
    
    # Agent Status Dashboard
    st.subheader("🔄 Real-Time Agent Status")
    
    # Agent cards in a grid
    cols = st.columns(4)
    
    agent_details = {
        "Ordering Agent": {
            "icon": "🛒",
            "description": "Processes customer interactions using NLP",
            "tools": ["LangChain LLM", "Conversation Memory", "Cart Manager"],
            "status": st.session_state.agents["Ordering Agent"]["status"]
        },
        "Safety Agent": {
            "icon": "🛡️",
            "description": "Validates prescriptions and checks interactions",
            "tools": ["Drug Database", "Interaction Checker", "Prescription Validator"],
            "status": st.session_state.agents["Safety Agent"]["status"]
        },
        "Forecast Agent": {
            "icon": "📈",
            "description": "Predicts inventory needs using ML models",
            "tools": ["Time Series Analysis", "Pattern Recognition", "Demand Forecasting"],
            "status": st.session_state.agents["Forecast Agent"]["status"]
        },
        "Procurement Agent": {
            "icon": "📦",
            "description": "Manages supplier interactions and POs",
            "tools": ["Supplier API", "Price Negotiator", "Order Manager"],
            "status": st.session_state.agents["Procurement Agent"]["status"]
        }
    }
    
    for idx, (agent_name, details) in enumerate(agent_details.items()):
        with cols[idx]:
            status_color = "green" if details["status"] == "Active" else "orange"
            
            st.markdown(f"""
            <div class="agent-card">
                <h3>{details['icon']} {agent_name}</h3>
                <p><strong>Status:</strong> <span style="color:{status_color}">{details['status']}</span></p>
                <p>{details['description']}</p>
                <p><small><strong>Tools:</strong> {', '.join(details['tools'][:2])}...</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Agent controls
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(f"Restart", key=f"restart_{agent_name}", use_container_width=True):
                    st.session_state.agents[agent_name]["status"] = "Active"
                    st.session_state.agents[agent_name]["last_active"] = datetime.now()
                    st.success(f"{agent_name} restarted!")
                    st.rerun()
            
            with col_b:
                if st.button(f"Logs", key=f"logs_{agent_name}", use_container_width=True):
                    st.info(f"Showing logs for {agent_name}...")
    
    # Agent Communication Graph
    st.subheader("🔄 Agent Communication Flow")
    
    st.markdown("""
    ```mermaid
    graph TD
        A[Customer Request] --> B{Ordering Agent}
        B --> C[Safety Agent]
        C --> D{Prescription Valid?}
        D -->|Yes| E[Forecast Agent]
        D -->|No| F[Request Prescription]
        E --> G[Analyze Consumption]
        G --> H[Generate Predictions]
        H --> I[Procurement Agent]
        I --> J[Supplier APIs]
        J --> K[Update Inventory]
        K --> L[Notify Customer]
        L --> M[End Workflow]
        
        style A fill:#e1f5fe
        style B fill:#f3e5f5
        style C fill:#fff3e0
        style I fill:#e8f5e8
        style L fill:#fce4ec
    ```
    """)
    
    # Workflow Simulation
    st.subheader("⚙️ Advanced Workflow Simulation")
    
    workflow_type = st.selectbox(
        "Select Workflow Scenario",
        [
            "Standard Order Processing",
            "Emergency Refill Request", 
            "Prescription Verification Flow",
            "Inventory Crisis Management",
            "Multi-Supplier Procurement",
            "Customer Complaint Resolution"
        ]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        simulation_speed = st.slider("Simulation Speed", 0.1, 2.0, 1.0, 0.1)
    
    with col2:
        include_mcp = st.checkbox("Include MCP Integration", value=True)
    
    if st.button("🚀 Execute Full Agent Workflow", type="primary", use_container_width=True):
        # Create progress tracking
        progress_bar = st.progress(0)
        status_container = st.empty()
        log_container = st.empty()
        
        # Simulate workflow steps
        steps = [
            ("Initializing LangGraph agent coordinator...", 5),
            ("Loading conversation memory from vector store...", 10),
            ("Ordering Agent parsing customer intent...", 20),
            ("Safety Agent querying prescription database...", 35),
            ("Checking drug interactions with knowledge graph...", 50),
            ("Forecast Agent running time-series prediction...", 65),
            ("Procurement Agent contacting supplier APIs...", 80),
            ("MCP tools executing real-world actions...", 95),
            ("Workflow completed successfully! ✅", 100)
        ]
        
        detailed_logs = []
        
        for step_text, progress in steps:
            # Update progress
            progress_bar.progress(progress)
            status_container.info(f"⏳ {step_text}")
            
            # Add to logs
            detailed_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {step_text}")
            
            # Simulate processing time
            time.sleep(0.3 / simulation_speed)
        
        # Show completion
        status_container.success("✅ Agent workflow executed successfully!")
        
        # Show detailed logs
        with st.expander("📋 View Detailed Execution Logs"):
            for log in detailed_logs:
                st.code(log)
        
        # Show MCP integration results if enabled
        if include_mcp:
            st.subheader("🔗 MCP Integration Results")
            
            mcp_results = [
                ("Zapier", "✅ Sent order confirmation via email and SMS"),
                ("n8n", "✅ Triggered supplier API and updated inventory"),
                ("Mediloon CMS", "✅ Synchronized order data in real-time"),
                ("Webhooks", "✅ Updated external pharmacy systems"),
                ("Payment Gateway", "✅ Processed transaction securely")
            ]
            
            for service, result in mcp_results:
                col_a, col_b = st.columns([1, 4])
                with col_a:
                    st.write(service)
                with col_b:
                    st.success(result)
        
        # Generate performance report
        report = {
            "workflow_type": workflow_type,
            "execution_time": f"{len(steps) * 0.3 / simulation_speed:.1f}s",
            "agents_used": list(st.session_state.agents.keys()),
            "success_rate": "99.8%",
            "cost_savings": f"${random.randint(50, 200)}",
            "timestamp": datetime.now().isoformat()
        }
        
        st.download_button(
            "📊 Download Performance Report",
            json.dumps(report, indent=2),
            file_name=f"agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# ==================== PAGE: INVENTORY ====================

elif page == "Inventory":
    st.title("📦 Smart Inventory Management System")
    
    # Summary metrics
    total_value = sum(item.get("total_value", 0) for item in st.session_state.inventory)
    low_stock_count = sum(1 for item in st.session_state.inventory 
                         if item["current_stock"] < item["reorder_level"])
    out_of_stock = sum(1 for item in st.session_state.inventory 
                      if item["current_stock"] == 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total SKUs", len(st.session_state.inventory))
    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col3:
        st.metric("Low Stock", low_stock_count, delta="Needs attention" if low_stock_count > 0 else None)
    with col4:
        st.metric("Out of Stock", out_of_stock, delta="Critical" if out_of_stock > 0 else None)
    
    # Inventory management tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stock Overview", 
        "⚠️ Alerts", 
        "🔄 Reorder Management", 
        "🏢 Suppliers", 
        "📈 Analytics"
    ])
    
    with tab1:
        # Interactive inventory editor
        st.subheader("Real-Time Inventory Dashboard")
        
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            
            # Add calculated fields
            inv_df["stock_status"] = inv_df.apply(
                lambda row: (
                    "🔴 Out of Stock" if row["current_stock"] == 0 else
                    "🟠 Very Low" if row["current_stock"] < row["reorder_level"] * 0.5 else
                    "🟡 Low" if row["current_stock"] < row["reorder_level"] else
                    "🟢 Good" if row["current_stock"] < row["reorder_level"] * 3 else
                    "🔵 Excess"
                ), axis=1
            )
            
            inv_df["days_of_supply"] = inv_df.apply(
                lambda row: (
                    row["current_stock"] / MEDICINES.get(row["medicine"], {"refill_days": 30})["refill_days"] * 30
                    if row["current_stock"] > 0 else 0
                ), axis=1
            )
            
            # Editable dataframe
            edited_df = st.data_editor(
                inv_df[[
                    "medicine", "current_stock", "reorder_level", 
                    "stock_status", "days_of_supply", "supplier", "total_value"
                ]],
                column_config={
                    "medicine": st.column_config.TextColumn("Medicine", width="medium"),
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
                    "stock_status": st.column_config.TextColumn("Status", width="small"),
                    "days_of_supply": st.column_config.NumberColumn(
                        "Days of Supply",
                        format="%.1f days",
                        width="small"
                    ),
                    "total_value": st.column_config.NumberColumn(
                        "Value",
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
    
    with tab2:
        st.subheader("⚠️ Inventory Alerts")
        
        alerts = []
        
        # Check various alert conditions
        for item in st.session_state.inventory:
            if item["current_stock"] == 0:
                alerts.append({
                    "type": "🔴 Critical",
                    "message": f"{item['medicine']} is OUT OF STOCK",
                    "action": "Emergency order required"
                })
            elif item["current_stock"] < item["reorder_level"] * 0.5:
                alerts.append({
                    "type": "🟠 High",
                    "message": f"{item['medicine']} is VERY LOW ({item['current_stock']} left)",
                    "action": "Urgent reorder needed"
                })
            elif item["current_stock"] < item["reorder_level"]:
                alerts.append({
                    "type": "🟡 Medium",
                    "message": f"{item['medicine']} is LOW ({item['current_stock']} left)",
                    "action": "Schedule reorder"
                })
        
        # Check for expired stock (simulated)
        for item in st.session_state.inventory:
            if random.random() < 0.1:  # 10% chance of expiring soon
                days_to_expire = random.randint(7, 30)
                alerts.append({
                    "type": "🟣 Warning",
                    "message": f"{item['medicine']} expires in {days_to_expire} days",
                    "action": "Check batch and prioritize use"
                })
        
        if alerts:
            # Sort by severity
            severity_order = {"🔴 Critical": 0, "🟠 High": 1, "🟡 Medium": 2, "🟣 Warning": 3}
            alerts.sort(key=lambda x: severity_order.get(x["type"], 4))
            
            for alert in alerts:
                with st.expander(f"{alert['type']}: {alert['message']}"):
                    st.write(f"**Action Required:** {alert['action']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"Take Action", key=f"action_{alert['message'][:20]}"):
                            st.info(f"Action initiated for {alert['message'].split()[0]}")
                    with col_b:
                        if st.button(f"Dismiss", key=f"dismiss_{alert['message'][:20]}"):
                            st.success("Alert dismissed")
                            st.rerun()
        else:
            st.success("✅ No active alerts. All inventory levels are satisfactory.")
    
    with tab3:
        st.subheader("🔄 Automated Reorder Management")
        
        # Generate reorder suggestions
        suggestions = []
        for item in st.session_state.inventory:
            if item["current_stock"] < item["reorder_level"]:
                deficit = item["reorder_level"] - item["current_stock"]
                suggested_qty = max(item["reorder_level"] * 2, 30, deficit * 3)
                
                suggestions.append({
                    "Medicine": item["medicine"],
                    "Current": item["current_stock"],
                    "Reorder At": item["reorder_level"],
                    "Deficit": deficit,
                    "Supplier": item["supplier"],
                    "Lead Time": f"{item['lead_time_days']} days",
                    "Suggested Qty": suggested_qty,
                    "Est. Cost": f"${suggested_qty * item.get('unit_price', 0):.2f}"
                })
        
        if suggestions:
            st.warning(f"⚠️ {len(suggestions)} items need reordering!")
            
            sugg_df = pd.DataFrame(suggestions)
            st.dataframe(sugg_df, use_container_width=True, hide_index=True)
            
            # Bulk actions
            st.subheader("📋 Batch Reorder Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📧 Generate All POs", use_container_width=True):
                    st.success(f"Generated {len(suggestions)} purchase orders!")
                    st.info("Purchase orders sent to Procurement Agent for processing.")
            
            with col2:
                selected_items = st.multiselect(
                    "Select items for manual reorder",
                    [s["Medicine"] for s in suggestions],
                    default=[s["Medicine"] for s in suggestions[:3]]
                )
                
                if st.button("🛒 Add to Cart", use_container_width=True):
                    quantities = []
                    for med in selected_items:
                        suggestion = next(s for s in suggestions if s["Medicine"] == med)
                        quantities.append(suggestion["Suggested Qty"])
                    
                    for med, qty in zip(selected_items, quantities):
                        add_to_cart_with_details(med, qty)
                    
                    st.success(f"Added {len(selected_items)} items to cart for reorder!")
                    st.rerun()
            
            with col3:
                if st.button("🤖 Auto-Order via Agents", type="primary", use_container_width=True):
                    # Simulate agent-based ordering
                    with st.spinner("Procurement Agent processing orders..."):
                        time.sleep(2)
                        
                        workflow = simulate_advanced_agent_workflow("inventory_replenishment")
                        st.success("Auto-ordering completed via agent system!")
                        
                        with st.expander("View Agent Workflow"):
                            for line in workflow:
                                st.code(line)
        else:
            st.success("✅ All stock levels are adequate. No reorders needed at this time.")
    
    with tab4:
        st.subheader("🏢 Supplier Management")
        
        # Group by supplier
        supplier_data = {}
        for item in st.session_state.inventory:
            supplier = item["supplier"]
            if supplier not in supplier_data:
                supplier_data[supplier] = {
                    "items": [],
                    "total_value": 0,
                    "last_order": item["last_ordered"],
                    "performance_score": random.randint(70, 100)
                }
            
            supplier_data[supplier]["items"].append(item["medicine"])
            supplier_data[supplier]["total_value"] += item.get("total_value", 0)
        
        # Display supplier info
        for supplier, data in supplier_data.items():
            with st.expander(f"🏭 {supplier} (Score: {data['performance_score']}/100)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Items Supplied:** {len(data['items'])}")
                    st.write(f"**Inventory Value:** ${data['total_value']:,.2f}")
                    st.write(f"**Last Order:** {data['last_order'].strftime('%Y-%m-%d')}")
                
                with col2:
                    # Performance indicators
                    st.write("**Performance:**")
                    st.progress(data["performance_score"] / 100, 
                               text=f"{data['performance_score']}%")
                    
                    if data["performance_score"] >= 90:
                        st.success("Excellent performance")
                    elif data["performance_score"] >= 80:
                        st.info("Good performance")
                    else:
                        st.warning("Needs improvement")
                
                # Items list
                st.write(f"**Items:** {', '.join(data['items'][:5])}{'...' if len(data['items']) > 5 else ''}")
                
                # Supplier actions
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if st.button(f"📧 Contact", key=f"contact_{supplier}"):
                        st.info(f"Opening contact form for {supplier}...")
                with col_b:
                    if st.button(f"📊 History", key=f"history_{supplier}"):
                        st.info(f"Loading order history for {supplier}...")
                with col_c:
                    if st.button(f"🔄 Reorder All", key=f"reorder_{supplier}"):
                        st.info(f"Initiating bulk reorder from {supplier}...")
    
    with tab5:
        st.subheader("📈 Inventory Analytics")
        
        if st.session_state.inventory:
            inv_df = pd.DataFrame(st.session_state.inventory)
            
            # Category analysis
            st.write("### 📊 Stock by Category")
            category_summary = inv_df.groupby("category").agg({
                "medicine": "count",
                "current_stock": "sum",
                "total_value": "sum"
            }).reset_index()
            
            category_summary.columns = ["Category", "SKU Count", "Total Stock", "Total Value"]
            st.dataframe(category_summary, use_container_width=True, hide_index=True)
            
            # ABC Analysis (Pareto)
            st.write("### 📈 ABC Analysis (Pareto)")
            
            # Sort by value
            value_df = inv_df.sort_values("total_value", ascending=False)
            value_df["cumulative_value"] = value_df["total_value"].cumsum()
            value_df["cumulative_percent"] = value_df["cumulative_value"] / value_df["total_value"].sum() * 100
            
            # Classify ABC
            value_df["abc_class"] = value_df["cumulative_percent"].apply(
                lambda x: "A" if x <= 80 else "B" if x <= 95 else "C"
            )
            
            # Show results
            abc_summary = value_df.groupby("abc_class").agg({
                "medicine": "count",
                "total_value": "sum"
            }).reset_index()
            
            st.dataframe(abc_summary, use_container_width=True, hide_index=True)
            
            # Turnover analysis
            st.write("### 🔄 Stock Turnover Analysis")
            
            # Simulate turnover data
            turnover_data = []
            for item in st.session_state.inventory:
                avg_monthly_sales = random.randint(5, 50)
                current_stock = item["current_stock"]
                
                if avg_monthly_sales > 0:
                    months_of_stock = current_stock / avg_monthly_sales
                    turnover_rate = avg_monthly_sales * 12 / current_stock if current_stock > 0 else 0
                    
                    turnover_data.append({
                        "Medicine": item["medicine"],
                        "Avg Monthly Sales": avg_monthly_sales,
                        "Current Stock": current_stock,
                        "Months of Stock": round(months_of_stock, 1),
                        "Turnover Rate": round(turnover_rate, 1)
                    })
            
            if turnover_data:
                turnover_df = pd.DataFrame(turnover_data)
                st.dataframe(turnover_df, use_container_width=True, hide_index=True)
                
                # Highlight slow movers
                slow_movers = turnover_df[turnover_df["Turnover Rate"] < 1]
                if not slow_movers.empty:
                    st.warning(f"⚠️ {len(slow_movers)} slow-moving items detected")
                    for _, row in slow_movers.iterrows():
                        st.write(f"• {row['Medicine']}: Turnover rate {row['Turnover Rate']}")

# ==================== PAGE: WORKFLOW ====================

elif page == "Workflow":
    st.title("⚙️ MCP & Workflow Automation")
    
    st.markdown("""
    ### Multiple Connection Platform (MCP) Integration
    
    TrackFusion 3 integrates with external tools and services for end-to-end automation 
    of pharmacy operations using AI agents and workflow engines.
    """)
    
    # MCP Services Status
    st.subheader("🔗 MCP Service Status")
    
    mcp_services = {
        "Zapier": {
            "status": "✅ Connected",
            "description": "Automates notifications across 5000+ apps",
            "usage": "High",
            "last_used": "5 minutes ago",
            "actions": ["Email", "SMS", "WhatsApp", "Slack", "CRM Updates"]
        },
        "n8n": {
            "status": "✅ Connected",
            "description": "Workflow automation and API integrations",
            "usage": "High",
            "last_used": "2 minutes ago",
            "actions": ["Supplier APIs", "Payment Processing", "Inventory Sync", "Data Transformation"]
        },
        "Mediloon CMS": {
            "status": "✅ Connected",
            "description": "Central pharmacy management system",
            "usage": "Continuous",
            "last_used": "Real-time",
            "actions": ["Inventory Management", "Order Processing", "Customer Database", "Reporting"]
        },
        "Webhooks": {
            "status": "✅ Active",
            "description": "Real-time event notifications",
            "usage": "Medium",
            "last_used": "10 minutes ago",
            "actions": ["System Updates", "External API Calls", "Database Sync", "Alert Triggers"]
        },
        "Supplier APIs": {
            "status": "⚠️ Partial",
            "description": "Direct supplier integrations",
            "usage": "Medium",
            "last_used": "Yesterday",
            "actions": ["Stock Checks", "Order Placement", "Price Updates", "Delivery Tracking"]
        }
    }
    
    # Display service cards
    cols = st.columns(len(mcp_services))
    for idx, (service, info) in enumerate(mcp_services.items()):
        with cols[idx]:
            status_color = "green" if "✅" in info["status"] else "orange" if "⚠️" in info["status"] else "red"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{service}</h4>
                <p><span style="color:{status_color}">{info['status']}</span></p>
                <p><small>{info['description']}</small></p>
                <p><small>Usage: {info['usage']}</small></p>
                <p><small>Last used: {info['last_used']}</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show actions on hover/click
            with st.expander("Actions", expanded=False):
                for action in info["actions"][:3]:
                    st.write(f"• {action}")
    
    # Workflow Designer
    st.subheader("🎨 Visual Workflow Designer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        workflow_type = st.selectbox(
            "Select Workflow Template",
            [
                "Standard Order Fulfillment",
                "Emergency Refill Process", 
                "Prescription Verification Flow",
                "Inventory Replenishment",
                "Customer Notification Chain",
                "Supplier Communication",
                "Payment Processing",
                "Returns & Refunds",
                "Custom Workflow"
            ]
        )
    
    with col2:
        trigger_type = st.selectbox(
            "Trigger Type",
            ["Manual", "Scheduled", "Event-Based", "API Call"]
        )
    
    # Workflow visualization
    st.markdown("### 🔄 Workflow Visualization")
    
    workflow_diagrams = {
        "Standard Order Fulfillment": """
        ```mermaid
        graph LR
            A[Customer Order] --> B[AI Assistant]
            B --> C[Prescription Check]
            C --> D{Valid?}
            D -->|Yes| E[Inventory Check]
            D -->|No| F[Request Prescription]
            E --> G{In Stock?}
            G -->|Yes| H[Process Payment]
            G -->|No| I[Supplier Order]
            H --> J[Confirm Order]
            I --> J
            J --> K[Update CMS]
            K --> L[Send Notifications]
            L --> M[Package & Ship]
            M --> N[Delivery Tracking]
            N --> O[Customer Feedback]
            
            style A fill:#e3f2fd
            style B fill:#f3e5f5
            style H fill:#e8f5e8
            style M fill:#fff3e0
        ```
        """,
        "Emergency Refill Process": """
        ```mermaid
        graph TD
            A[Low Stock Alert] --> B[Forecast Agent]
            B --> C[Urgency Assessment]
            C --> D{Critical?}
            D -->|Yes| E[Emergency Supplier Call]
            D -->|No| F[Standard Reorder]
            E --> G[Expedited Shipping]
            F --> H[Regular Shipping]
            G --> I[Priority Receiving]
            H --> J[Standard Receiving]
            I --> K[Update Inventory]
            J --> K
            K --> L[Notify Pharmacist]
            L --> M[Customer Notification]
            
            style A fill:#ffebee
            style E fill:#fff3e0
            style G fill:#e8f5e8
        ```
        """
    }
    
    st.markdown(workflow_diagrams.get(workflow_type, "Select a workflow to see visualization"))
    
    # Workflow Execution
    st.subheader("🚀 Execute Workflow")
    
    execution_tab1, execution_tab2, execution_tab3 = st.tabs(["Manual Execution", "Scheduled", "API Endpoint"])
    
    with execution_tab1:
        st.write("Execute workflow manually with custom parameters")
        
        # Execution parameters
        params = {}
        
        col_a, col_b = st.columns(2)
        with col_a:
            params["priority"] = st.selectbox("Priority", ["Normal", "High", "Emergency"])
            params["notify_customer"] = st.checkbox("Notify Customer", value=True)
            params["update_inventory"] = st.checkbox("Update Inventory", value=True)
        
        with col_b:
            params["generate_reports"] = st.checkbox("Generate Reports", value=True)
            params["log_activity"] = st.checkbox("Log Activity", value=True)
            params["test_mode"] = st.checkbox("Test Mode", value=False)
        
        if st.button("▶️ Execute Workflow Now", type="primary", use_container_width=True):
            # Simulate workflow execution
            progress_bar = st.progress(0)
            status_container = st.empty()
            log_container = st.empty()
            
            execution_steps = [
                ("Initializing workflow engine...", 10),
                ("Loading MCP connections...", 20),
                ("Authenticating with external services...", 30),
                ("Executing agent coordination...", 45),
                ("Processing workflow steps...", 60),
                ("Integrating with external systems...", 75),
                ("Updating databases and CMS...", 85),
                ("Sending notifications and alerts...", 95),
                ("Workflow execution complete! ✅", 100)
            ]
            
            execution_logs = []
            
            for step_text, progress in execution_steps:
                # Update progress
                progress_bar.progress(progress)
                status_container.info(f"⏳ {step_text}")
                
                # Add to logs
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                execution_logs.append(f"[{timestamp}] {step_text}")
                
                # Simulate processing time
                time.sleep(0.4 if not params["test_mode"] else 0.1)
            
            # Show completion
            status_container.success("✅ Workflow executed successfully!")
            
            # Show execution report
            with st.expander("📊 Execution Report", expanded=True):
                st.write("**Workflow Summary**")
                st.write(f"- **Type:** {workflow_type}")
                st.write(f"- **Priority:** {params['priority']}")
                st.write(f"- **Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- **Duration:** {len(execution_steps) * 0.4:.1f} seconds")
                st.write(f"- **Status:** Completed Successfully")
                st.write(f"- **Test Mode:** {'Yes' if params['test_mode'] else 'No'}")
                
                st.write("**MCP Integrations Used:**")
                for service, info in mcp_services.items():
                    if info["usage"] in ["High", "Continuous"]:
                        st.write(f"- {service}: {info['status']}")
            
            # Show agent workflow
            st.subheader("🤖 Agent Workflow Execution")
            workflow = simulate_advanced_agent_workflow()
            for line in workflow:
                if line.strip():
                    st.code(line)
            
            # Download execution logs
            st.download_button(
                "📥 Download Execution Logs",
                "\n".join(execution_logs),
                file_name=f"workflow_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with execution_tab2:
        st.write("Schedule automated workflow execution")
        
        schedule_type = st.radio(
            "Schedule Type",
            ["Daily", "Weekly", "Monthly", "Custom Cron"],
            horizontal=True
        )
        
        if schedule_type == "Daily":
            schedule_time = st.time_input("Execution Time", value=datetime.now().time())
            cron_expr = f"0 {schedule_time.hour} {schedule_time.minute} * * *"
        
        elif schedule_type == "Weekly":
            col1, col2 = st.columns(2)
            with col1:
                schedule_day = st.selectbox("Day of Week", 
                                          ["Monday", "Tuesday", "Wednesday", 
                                           "Thursday", "Friday", "Saturday", "Sunday"])
            with col2:
                schedule_time = st.time_input("Execution Time", value=datetime.now().time())
            
            day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                      "Friday", "Saturday", "Sunday"].index(schedule_day)
            cron_expr = f"0 {schedule_time.hour} {schedule_time.minute} * * {day_num}"
        
        elif schedule_type == "Monthly":
            schedule_day = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
            schedule_time = st.time_input("Execution Time", value=datetime.now().time())
            cron_expr = f"0 {schedule_time.hour} {schedule_time.minute} {schedule_day} * *"
        
        else:
            cron_expr = st.text_input("Cron Expression", value="0 9 * * *")
            st.caption("Format: minute hour day month day-of-week")
            st.caption("Example: 0 9 * * * = Daily at 9:00 AM")
        
        st.write(f"**Cron Expression:** `{cron_expr}`")
        
        enable_schedule = st.checkbox("Enable Schedule", value=True)
        
        if st.button("💾 Save Schedule", use_container_width=True):
            if enable_schedule:
                st.success(f"✅ Workflow scheduled using: `{cron_expr}`")
                st.info("Scheduled workflows will run automatically in the background.")
                add_notification(
                    "Schedule Created",
                    f"Workflow '{workflow_type}' scheduled: {cron_expr}",
                    "medium"
                )
            else:
                st.warning("⚠️ Schedule saved but disabled. Enable to activate.")
    
    with execution_tab3:
        st.write("API Endpoint Configuration")
        
        st.code("""
POST /api/v1/workflow/execute
Content-Type: application/json
Authorization: Bearer <your_api_key>

{
  "workflow_type": "standard_order",
  "parameters": {
    "order_id": "ORD123456",
    "customer_id": "CUST789",
    "priority": "normal"
  },
  "callback_url": "https://your-callback-url.com/webhook"
}
        """, language="json")
        
        st.write("**Webhook Configuration**")
        
        webhook_url = st.text_input("Callback URL", placeholder="https://your-domain.com/webhook")
        api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
        
        if st.button("Test Webhook", use_container_width=True):
            with st.spinner("Sending test webhook..."):
                time.sleep(1)
                st.success("✅ Webhook test successful!")
                st.info("Check your server logs for the incoming request.")

# ==================== PAGE: SETTINGS ====================

elif page == "Settings":
    st.title("⚙️ System Configuration")
    
    settings_tabs = st.tabs(["General", "AI & Agents", "Notifications", 
                            "API Integration", "Privacy", "System Info"])
    
    with settings_tabs[0]:
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Language settings
            language = st.selectbox(
                "Interface Language",
                list(LANGUAGES.keys()),
                index=list(LANGUAGES.keys()).index(st.session_state.system_settings.get('language', 'English'))
            )
            
            # Timezone
            timezone = st.selectbox(
                "Timezone",
                ["UTC", "EST (UTC-5)", "PST (UTC-8)", "CET (UTC+1)", 
                 "IST (UTC+5:30)", "AEST (UTC+10)", "JST (UTC+9)"],
                index=0
            )
            
            # Date format
            date_format = st.selectbox(
                "Date Format",
                ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY", "Month DD, YYYY"],
                index=0
            )
            
            # Currency
            currency = st.selectbox(
                "Currency",
                ["USD ($)", "EUR (€)", "GBP (£)", "AED (د.إ)", "INR (₹)", "JPY (¥)"],
                index=0
            )
        
        with col2:
            # Feature toggles
            st.write("**Feature Toggles**")
            
            auto_refill = st.checkbox(
                "Enable Auto-Refill", 
                value=st.session_state.system_settings.get('auto_refill', True),
                help="Automatically refill medicines based on predictions"
            )
            
            voice_enabled = st.checkbox(
                "Enable Voice Input", 
                value=st.session_state.system_settings.get('voice_enabled', False),
                help="Allow voice commands for ordering"
            )
            
            dark_mode = st.checkbox(
                "Dark Mode", 
                value=st.session_state.system_settings.get('dark_mode', False),
                help="Switch to dark theme"
            )
            
            advanced_analytics = st.checkbox(
                "Advanced Analytics", 
                value=True,
                help="Enable detailed analytics and insights"
            )
            
            # Data retention
            st.write("**Data Retention**")
            retention_days = st.slider("Retain Data For (days)", 30, 730, 90)
            auto_backup = st.checkbox("Automatic Daily Backup", value=True)
            backup_location = st.selectbox(
                "Backup Location",
                ["Cloud Storage", "Local Server", "Both"],
                index=0
            )
        
        if st.button("💾 Save General Settings", use_container_width=True):
            st.session_state.system_settings.update({
                'language': language,
                'auto_refill': auto_refill,
                'voice_enabled': voice_enabled,
                'dark_mode': dark_mode
            })
            st.success("General settings saved successfully!")
    
    with settings_tabs[1]:
        st.subheader("AI & Agent Configuration")
        
        # AI Model Settings
        st.write("**AI Model Configuration**")
        
        ai_provider = st.radio(
            "AI Provider",
            ["OpenAI GPT-4", "Anthropic Claude 3", "Google Gemini Pro", 
             "Local LLM", "Hybrid Mode"],
            horizontal=True,
            index=0
        )
        
        if ai_provider in ["OpenAI GPT-4", "Anthropic Claude 3", "Google Gemini Pro"]:
            col_a, col_b = st.columns(2)
            with col_a:
                temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
                max_tokens = st.number_input("Max Response Tokens", 100, 4000, 1000)
            with col_b:
                frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
                presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1)
        elif ai_provider == "Local LLM":
            model_name = st.selectbox(
                "Local Model",
                ["Llama 2 7B", "Mistral 7B", "Phi-2", "Custom Model"]
            )
            gpu_acceleration = st.checkbox("GPU Acceleration", value=True)
            quantized = st.checkbox("Use Quantized Model", value=True)
        else:  # Hybrid Mode
            st.write("**Hybrid Configuration**")
            primary_model = st.selectbox("Primary Model", ["GPT-4", "Claude 3", "Gemini Pro"])
            fallback_model = st.selectbox("Fallback Model", ["Local LLM", "Rules Engine"])
            fallback_threshold = st.slider("Fallback Confidence Threshold", 0.5, 1.0, 0.7, 0.05)
        
        # Agent Configuration
        st.write("**Agent System Configuration**")
        
        agent_autonomy = st.slider(
            "Agent Autonomy Level", 
            0, 100, 75,
            help="Higher values allow agents to make more decisions without human confirmation"
        )
        
        decision_timeout = st.number_input("Decision Timeout (seconds)", 1, 300, 30)
        max_retries = st.number_input("Max Retry Attempts", 1, 10, 3)
        
        col1, col2 = st.columns(2)
        with col1:
            enable_agent_logging = st.checkbox("Enable Agent Logging", value=True)
            enable_agent_learning = st.checkbox("Enable Agent Learning", value=True)
            allow_parallel_execution = st.checkbox("Allow Parallel Execution", value=True)
        with col2:
            allow_human_override = st.checkbox("Allow Human Override", value=True)
            enable_failover = st.checkbox("Enable Failover Mode", value=True)
            notify_on_failover = st.checkbox("Notify on Failover", value=True)
        
        if st.button("💾 Save AI Settings", use_container_width=True):
            st.success("AI and Agent settings saved successfully!")
    
    with settings_tabs[2]:
        st.subheader("Notification Settings")
        
        st.write("**Notification Channels**")
        
        notification_cols = st.columns(3)
        with notification_cols[0]:
            email_notifications = st.checkbox("Email Notifications", value=True)
            order_confirmations = st.checkbox("Order Confirmations", value=True)
            shipping_updates = st.checkbox("Shipping Updates", value=True)
            payment_receipts = st.checkbox("Payment Receipts", value=True)
        
        with notification_cols[1]:
            sms_notifications = st.checkbox("SMS Notifications", value=True)
            refill_reminders = st.checkbox("Refill Reminders", value=True)
            prescription_alerts = st.checkbox("Prescription Alerts", value=True)
            low_stock_alerts = st.checkbox("Low Stock Alerts", value=True)
        
        with notification_cols[2]:
            whatsapp_messages = st.checkbox("WhatsApp Messages", value=False)
            push_notifications = st.checkbox("Push Notifications", value=True)
            system_alerts = st.checkbox("System Alerts", value=True)
            marketing_emails = st.checkbox("Marketing Emails", value=False)
        
        st.write("**Notification Preferences**")
        
        immediate_notifications = st.checkbox("Immediate Notifications", value=True)
        daily_summary = st.checkbox("Daily Summary", value=False)
        weekly_report = st.checkbox("Weekly Report", value=True)
        monthly_insights = st.checkbox("Monthly Insights", value=True)
        
        quiet_hours = st.checkbox("Enable Quiet Hours", value=False)
        if quiet_hours:
            quiet_start, quiet_end = st.slider(
                "Quiet Hours",
                value=(22, 8),
                format="%H:00",
                help="No notifications will be sent during these hours"
            )
        
        notification_language = st.selectbox(
            "Notification Language",
            ["Same as interface", "English", "German", "Arabic", "Customer Preference"],
            index=0
        )
        
        if st.button("💾 Save Notification Settings", use_container_width=True):
            st.success("Notification settings saved successfully!")
    
    with settings_tabs[3]:
        st.subheader("API Integration Settings")
        
        st.warning("⚠️ API keys are sensitive information. Store them securely using environment variables in production.")
        
        # MCP API Keys
        st.write("**MCP Integration Keys**")
        
        zapier_key = st.text_input("Zapier API Key", type="password", 
                                  placeholder="zapier_xxxxxxxxxxxxxxxxxxxxxxxx")
        n8n_webhook = st.text_input("n8n Webhook URL", type="password",
                                   placeholder="https://hook.n8n.cloud/xxxxxxxx")
        mediloon_api = st.text_input("Mediloon CMS API Key", type="password",
                                    placeholder="ml_xxxxxxxxxxxxxxxxxxxxxxxx")
        
        # External Service Keys
        with st.expander("External Service API Keys"):
            openai_key = st.text_input("OpenAI API Key", type="password")
            anthropic_key = st.text_input("Anthropic API Key", type="password")
            google_api_key = st.text_input("Google Cloud API Key", type="password")
            twilio_sid = st.text_input("Twilio Account SID", type="password")
            twilio_auth = st.text_input("Twilio Auth Token", type="password")
        
        # Webhook Configuration
        with st.expander("Webhook Configuration"):
            webhook_secret = st.text_input("Webhook Secret", type="password")
            webhook_url = st.text_input("Webhook URL", 
                                       placeholder="https://your-domain.com/webhook")
            webhook_events = st.multiselect(
                "Webhook Events",
                ["order.created", "order.updated", "inventory.low", 
                 "prescription.verified", "refill.reminder", "payment.processed"],
                default=["order.created", "inventory.low"]
            )
        
        # Test connections
        if st.button("🔗 Test API Connections", use_container_width=True):
            with st.spinner("Testing API connections..."):
                time.sleep(2)
                
                test_results = []
                
                # Simulate connection tests
                if zapier_key:
                    test_results.append(("Zapier", "✅ Connected", "success"))
                else:
                    test_results.append(("Zapier", "❌ Not configured", "error"))
                
                if n8n_webhook:
                    test_results.append(("n8n", "✅ Connected", "success"))
                else:
                    test_results.append(("n8n", "❌ Not configured", "error"))
                
                if mediloon_api:
                    test_results.append(("Mediloon CMS", "✅ Connected", "success"))
                else:
                    test_results.append(("Mediloon CMS", "❌ Not configured", "error"))
                
                # Display results
                for name, status, result_type in test_results:
                    if result_type == "success":
                        st.success(f"{name}: {status}")
                    else:
                        st.error(f"{name}: {status}")
        
        if st.button("💾 Save API Keys", use_container_width=True):
            st.success("API keys saved successfully!")
            st.info("Note: Keys are stored in session memory only. For production, use environment variables or a secure vault.")
    
    with settings_tabs[4]:
        st.subheader("Privacy & Data Settings")
        
        st.write("**Data Privacy**")
        
        data_collection = st.checkbox(
            "Allow anonymous data collection for improvement", 
            value=True,
            help="Help us improve by sharing anonymous usage data"
        )
        
        personalized_ads = st.checkbox(
            "Allow personalized recommendations", 
            value=True,
            help="Show personalized medicine recommendations based on your history"
        )
        
        share_health_data = st.checkbox(
            "Share health data with healthcare providers (with consent)", 
            value=False,
            help="Allow sharing of relevant health data with your doctors"
        )
        
        auto_delete_data = st.checkbox(
            "Auto-delete old data after 2 years", 
            value=True
        )
        
        st.write("**Export Your Data**")
        
        if st.button("Export All My Data", use_container_width=True):
            export_data = {
                "user_profile": st.session_state.user_profile,
                "orders": st.session_state.orders,
                "inventory": st.session_state.inventory,
                "settings": st.session_state.system_settings,
                "exported_at": datetime.now().isoformat()
            }
            
            st.download_button(
                label="📥 Download Your Data",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"trackfusion_data_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        st.write("**Delete Account**")
        
        if st.button("Request Account Deletion", type="secondary", use_container_width=True):
            st.warning("Are you sure you want to delete your account? This action cannot be undone.")
            if st.button("Confirm Account Deletion", type="primary"):
                st.error("Account deletion requested. This will be processed within 30 days.")
    
    with settings_tabs[5]:
        st.subheader("System Information")
        
        # System metrics
        st.write("**Application Information**")
        
        info_cols = st.columns(3)
        with info_cols[0]:
            st.write("**Version:** TrackFusion 3.0.1")
            st.write("**Build Date:** 2024-01-15")
            st.write("**License:** MIT Open Source")
        
        with info_cols[1]:
            st.write(f"**Python Version:** {sys.version.split()[0]}")
            st.write("**Streamlit:** 1.29.0+")
            st.write("**Environment:** HackFusion Demo")
        
        with info_cols[2]:
            st.write(f"**Data Storage:** Session Memory")
            st.write(f"**AI Models:** Simulated")
            st.write(f"**Active Agents:** {sum(1 for a in st.session_state.agents.values() if a['status'] == 'Active')}/4")
        
        # System status
        st.write("**System Status**")
        
        status_items = [
            ("AI Assistant", "✅ Operational"),
            ("Predictive Engine", "✅ Operational"),
            ("Agent System", "✅ Operational"),
            ("Inventory Management", "✅ Operational"),
            ("Workflow Automation", "✅ Operational"),
            ("External APIs", "🟡 Limited"),
            ("Database", "✅ Connected"),
            ("Notifications", "✅ Active")
        ]
        
        for name, status in status_items:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(name)
            with col_b:
                if "✅" in status:
                    st.success(status)
                elif "🟡" in status:
                    st.warning(status)
                else:
                    st.error(status)
        
        # System actions
        st.write("**System Actions**")
        
        action_cols = st.columns(3)
        
        with action_cols[0]:
            if st.button("🔄 Clear Cache", use_container_width=True):
                keys_to_keep = ['system_settings', 'onboarding_complete', 'user_profile']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                init_session_state()
                init_demo_data()
                st.success("Cache cleared! Demo data reinitialized.")
                st.rerun()
        
        with action_cols[1]:
            if st.button("📊 Generate System Report", use_container_width=True):
                report = {
                    "system_status": {
                        "orders_count": len(st.session_state.orders),
                        "inventory_count": len(st.session_state.inventory),
                        "agents_status": st.session_state.agents,
                        "user_profile": st.session_state.user_profile.get('name', 'Not set')
                    },
                    "performance_metrics": {
                        "prediction_accuracy": f"{random.randint(85, 98)}%",
                        "order_processing_time": f"{random.randint(1, 5)} seconds",
                        "inventory_accuracy": f"{random.randint(95, 100)}%",
                        "customer_satisfaction": f"{random.randint(4, 5)}/5"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📥 Download System Report",
                    data=json.dumps(report, indent=2),
                    file_name="system_report.json",
                    mime="application/json"
                )
        
        with action_cols[2]:
            if st.button("🔄 Restart System", use_container_width=True):
                st.warning("System restart initiated...")
                time.sleep(1)
                st.success("System restarted successfully!")
                st.rerun()

# ==================== FOOTER ====================

# Language selector in footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 HackFusion Project")
st.sidebar.markdown("**TrackFusion 3** - AI-Driven Autonomous Pharmacy")
st.sidebar.markdown("---")

# Add version and copyright
st.sidebar.caption(f"v3.0.1 • {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("© 2024 Mediloon. All rights reserved.")

# Initialize on first run
if __name__ == "__main__":
    # Ensure data is initialized
    if not st.session_state.orders:
        init_demo_data()
