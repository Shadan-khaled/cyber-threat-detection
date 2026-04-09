

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import hashlib
import requests


# ENUMS والتعريفات

class LLMProvider(Enum):
    """مزودو نماذج اللغة"""
    GEMMA = "gemma"              
    LLAMA = "llama"              
    MISTRAL = "mistral"          
    ANTHROPIC = "anthropic"      
    OPENAI = "openai"            


class IntegrationType(Enum):
    """أنواع التكامل"""
    CRM = "crm"
    ERP = "erp"
    KNOWLEDGE_BASE = "kb"
    ANALYTICS = "analytics"
    DATABASE = "database"
    API = "api"


class AgentRole(Enum):
    """أدوار الوكلاء"""
    ANALYST = "analyst"
    CUSTOMER_SERVICE = "support"
    SALES = "sales"
    HR = "hr"
    RESEARCH = "research"



# إدارة نموذج GEMMA


class GEMMAModelManager:
    """مدير نموذج GEMMA 2B المجاني"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "gemma:2b"
        self.logger = logging.getLogger(__name__)
        self.is_running = False
    
    async def check_ollama_installed(self) -> bool:
        """التحقق من تثبيت Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.logger.info("✓ Ollama is installed")
                return True
            return False
        except Exception as e:
            self.logger.warning(f"Ollama not found: {str(e)}")
            return False
    
    async def start_ollama(self) -> bool:
        """بدء خادم Ollama"""
        try:
            # التحقق من أن Ollama يعمل
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                self.logger.info("✓ Ollama is already running")
                self.is_running = True
                return True
        except:
            pass
        
        # محاولة بدء Ollama
        try:
            self.logger.info("Starting Ollama service...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await asyncio.sleep(3)
            self.is_running = True
            self.logger.info("✓ Ollama started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Ollama: {str(e)}")
            return False
    
    async def download_gemma(self) -> bool:
        """تحميل نموذج GEMMA 2B"""
        try:
            self.logger.info(f"Downloading {self.model_name}...")
            
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                timeout=300
            )
            
            if response.status_code == 200:
                self.logger.info(f"✓ {self.model_name} downloaded successfully")
                return True
            else:
                self.logger.error(f"Failed to download model: {response.text}")
                return False
        
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False
    
    async def query(self, prompt: str) -> str:
        """الاستعلام من نموذج GEMMA"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                self.logger.error(f"Query failed: {response.text}")
                return ""
        
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            return ""
    
    async def setup(self) -> bool:
        """إعداد GEMMA"""
        self.logger.info("Setting up GEMMA 2B...")
        
        # التحقق من التثبيت
        if not await self.check_ollama_installed():
            self.logger.error("Ollama not installed. Please install from https://ollama.ai")
            return False
        
        # بدء الخدمة
        if not await self.start_ollama():
            self.logger.error("Failed to start Ollama")
            return False
        
        # تحميل النموذج
        if not await self.download_gemma():
            self.logger.error("Failed to download GEMMA")
            return False
        
        self.logger.info("✓ GEMMA setup completed")
        return True


# مكونات النظام الهجين مع GEMMA

class HybridLLMBridge:
    """جسر التكامل بين نماذج اللغة (مع GEMMA كالأساسي)"""
    
    def __init__(self):
        self.gemma_manager = GEMMAModelManager()
        self.paid_providers: Dict[str, Dict[str, str]] = {}
        self.active_provider: LLMProvider = LLMProvider.GEMMA
        self.fallback_chain = [
            LLMProvider.GEMMA,      # الأساسي (مجاني)
            LLMProvider.LLAMA,      # احتياطي (مجاني)
            LLMProvider.MISTRAL,    # احتياطي (مجاني)
        ]
        self.logger = logging.getLogger(__name__)
    
    async def setup(self) -> bool:
        """إعداد النموذج"""
        self.logger.info("Setting up LLM Bridge with GEMMA...")
        return await self.gemma_manager.setup()
    
    def register_paid_provider(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str
    ) -> None:
        """تسجيل موفر مدفوع (احتياطي)"""
        self.paid_providers[provider.value] = {
            "api_key": api_key,
            "model": model
        }
        self.logger.info(f"Registered {provider.value} as fallback")
    
    async def query(self, prompt: str) -> str:
        """الاستعلام من النموذج"""
        # المحاولة الأولى: GEMMA (مجاني)
        if self.active_provider == LLMProvider.GEMMA:
            response = await self.gemma_manager.query(prompt)
            if response:
                self.logger.info("✓ Query executed on GEMMA")
                return response
            else:
                self.logger.warning("GEMMA query failed, trying fallback...")
        
        # البدائل (النماذج المدفوعة)
        for provider in self.fallback_chain[1:]:
            if provider.value in self.paid_providers:
                self.logger.info(f"Trying {provider.value} as fallback...")
                # هنا نضع منطق استعلام المزود المدفوع
                return f"Response from {provider.value}"
        
        return "Error: No available LLM providers"


class LocalKnowledgeBase:
    """قاعدة المعرفة المحلية"""
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """إضافة مستند"""
        self.documents[doc_id] = {
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(doc_id)
        
        self.logger.info(f"Document '{doc_id}' added")
    
    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """البحث في قاعدة المعرفة"""
        results = []
        query_lower = query.lower()
        
        docs_to_search = self.documents.items()
        if category and category in self.categories:
            doc_ids = self.categories[category]
            docs_to_search = [
                (doc_id, self.documents[doc_id])
                for doc_id in doc_ids
            ]
        
        for doc_id, doc in docs_to_search:
            if query_lower in doc["content"].lower():
                results.append({
                    "doc_id": doc_id,
                    "content": doc["content"][:200],
                    "category": doc["category"],
                    "full_content": doc["content"]
                })
        
        return results[:limit]


class ExternalIntegration:
    """تكامل مع الأنظمة الخارجية"""
    
    def __init__(self, integration_type: IntegrationType):
        self.type = integration_type
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, config: Dict[str, str]) -> bool:
        """الاتصال بالنظام الخارجي"""
        self.config = config
        self.logger.info(f"Connected to {self.type.value}")
        return True
    
    async def query(self, query: str) -> Dict[str, Any]:
        """الاستعلام من النظام الخارجي"""
        return {
            "system": self.type.value,
            "query": query,
            "results": f"Results from {self.type.value}"
        }


class HybridAgent:
    """وكيل ذكي هجين مع GEMMA"""
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm_bridge: HybridLLMBridge,
        kb: LocalKnowledgeBase
    ):
        self.name = name
        self.role = role
        self.llm_bridge = llm_bridge
        self.kb = kb
        self.integrations: Dict[str, ExternalIntegration] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def register_integration(
        self,
        integration_type: IntegrationType,
        config: Dict[str, str]
    ) -> None:
        """تسجيل تكامل خارجي"""
        integration = ExternalIntegration(integration_type)
        asyncio.run(integration.connect(config))
        self.integrations[integration_type.value] = integration
    
    async def execute(self, query: str) -> Dict[str, Any]:
        """تنفيذ المهمة"""
        execution_id = f"{self.name}_{datetime.now().timestamp()}"
        
        self.logger.info(f"Executing: {query}")
        
        # المرحلة 1: تعزيز الاستعلام
        search_results = self.kb.search(query)
        augmented_query = query
        if search_results:
            context = "\n".join([r["full_content"][:100] for r in search_results])
            augmented_query = f"{query}\n\nContext: {context}"
        
        # المرحلة 2: استعلام GEMMA
        llm_response = await self.llm_bridge.query(augmented_query)
        
        # المرحلة 3: جلب البيانات الخارجية
        external_data = {}
        for int_type, integration in self.integrations.items():
            result = await integration.query(query)
            external_data[int_type] = result
        
        # النتيجة النهائية
        result = {
            "execution_id": execution_id,
            "agent": self.name,
            "role": self.role.value,
            "query": query,
            "llm_response": llm_response,
            "external_data": external_data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات"""
        return {
            "name": self.name,
            "role": self.role.value,
            "executions": len(self.execution_history),
            "integrations": list(self.integrations.keys())
        }


class EnterpriseAgentManager:
    """مدير الوكلاء للمؤسسات"""
    
    def __init__(self):
        self.agents: Dict[str, HybridAgent] = {}
        self.llm_bridge = HybridLLMBridge()
        self.kb = LocalKnowledgeBase()
        self.logger = logging.getLogger(__name__)
    
    async def setup(self) -> bool:
        """إعداد النظام"""
        self.logger.info("Setting up Enterprise Agent Manager...")
        return await self.llm_bridge.setup()
    
    def register_paid_provider(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str
    ) -> None:
        """تسجيل موفر مدفوع (احتياطي)"""
        self.llm_bridge.register_paid_provider(provider, api_key, model)
    
    def create_agent(
        self,
        agent_id: str,
        name: str,
        role: AgentRole
    ) -> HybridAgent:
        """إنشاء وكيل جديد"""
        agent = HybridAgent(name, role, self.llm_bridge, self.kb)
        self.agents[agent_id] = agent
        self.logger.info(f"Agent created: {agent_id}")
        return agent
    
    def add_knowledge(
        self,
        doc_id: str,
        content: str,
        category: str
    ) -> None:
        """إضافة معرفة"""
        self.kb.add_document(doc_id, content, category)
    
    async def execute_agent(self, agent_id: str, query: str) -> Dict[str, Any]:
        """تنفيذ وكيل"""
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}
        
        agent = self.agents[agent_id]
        return await agent.execute(query)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام"""
        return {
            "total_agents": len(self.agents),
            "agents": {
                agent_id: agent.get_stats()
                for agent_id, agent in self.agents.items()
            },
            "primary_model": "GEMMA 2B (Free)",
            "knowledge_base_categories": list(self.kb.categories.keys())
        }


# ============================================================
# مثال على الاستخدام
# ============================================================

async def demo():
    """عرض توضيحي"""
    print("\n" + "="*60)
    print("🤖 نظام الوكلاء الهجين مع GEMMA 2B (مجاني)")
    print("="*60 + "\n")
    
    # إنشاء المدير
    manager = EnterpriseAgentManager()
    
    # الإعداد
    print("⚙️ إعداد GEMMA 2B...")
    setup_success = await manager.setup()
    
    if setup_success:
        print("✅ GEMMA 2B جاهز!\n")
    else:
        print("❌ فشل إعداد GEMMA. يرجى تثبيت Ollama من https://ollama.ai\n")
        return
    
    # إضافة معرفة
    print("📚 إضافة قاعدة المعرفة...")
    manager.add_knowledge(
        "policy_001",
        "سياسة الإجازات: 20 يوم سنوياً",
        "hr"
    )
    manager.add_knowledge(
        "product_001",
        "المنتج الأساسي بسعر 99 ريال",
        "sales"
    )
    print("✅ تمت إضافة المعرفة\n")
    
    # إنشاء وكلاء
    print("🤖 إنشاء الوكلاء...")
    support_agent = manager.create_agent(
        "support_bot",
        "Support Assistant",
        AgentRole.CUSTOMER_SERVICE
    )
    sales_agent = manager.create_agent(
        "sales_bot",
        "Sales Assistant",
        AgentRole.SALES
    )
    print("✅ تم إنشاء الوكلاء\n")
    
    # تشغيل استعلامات
    print("💬 تشغيل الاستعلامات...\n")
    
    queries = {
        "support_bot": "كم عدد أيام الإجازة السنوية؟",
        "sales_bot": "ما سعر المنتج الأساسي؟"
    }
    
    for agent_id, query in queries.items():
        print(f"❓ Q: {query}")
        result = await manager.execute_agent(agent_id, query)
        print(f"✅ A: {result['llm_response']}\n")
    
    # الإحصائيات
    print("="*60)
    print("📊 إحصائيات النظام:")
    print("="*60)
    stats = manager.get_system_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n❌ تم إيقاف البرنامج")
    except Exception as e:
        print(f"\n❌ خطأ: {str(e)}")
