"""
Service Runner - Utility to run multiple services for development and testing.

Provides convenient way to start/stop all microservices for development
and testing scenarios.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional
import subprocess
import time
import os
from pathlib import Path

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class ServiceProcess:
    """Manages a single service process."""
    
    def __init__(self, name: str, module: str, port: int, env: Optional[Dict[str, str]] = None):
        self.name = name
        self.module = module
        self.port = port
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
    
    async def start(self) -> bool:
        """Start the service process."""
        if self.process and self.process.poll() is None:
            logger.warning(f"Service {self.name} is already running")
            return True
        
        try:
            logger.info(f"Starting {self.name} on port {self.port}...")
            
            # Prepare environment
            env = dict(os.environ)
            env.update(self.env)
            
            # Start process
            self.process = subprocess.Popen([
                sys.executable, "-m", self.module
            ], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.start_time = time.time()
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            # Check if it's still running
            if self.process.poll() is None:
                logger.info(f"✅ {self.name} started successfully (PID: {self.process.pid})")
                return True
            else:
                stdout, stderr = self.process.communicate()
                logger.error(f"❌ {self.name} failed to start")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start {self.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the service process."""
        if not self.process or self.process.poll() is not None:
            logger.warning(f"Service {self.name} is not running")
            return True
        
        try:
            logger.info(f"Stopping {self.name}...")
            
            # Send SIGTERM
            self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=10)
                logger.info(f"✅ {self.name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                logger.warning(f"Force killing {self.name}...")
                self.process.kill()
                self.process.wait()
                logger.info(f"✅ {self.name} force stopped")
            
            self.process = None
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop {self.name}: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self.process is not None and self.process.poll() is None
    
    def get_status(self) -> Dict[str, any]:
        """Get service status information."""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            "name": self.name,
            "running": self.is_running(),
            "pid": self.process.pid if self.process else None,
            "port": self.port,
            "uptime_seconds": uptime if self.is_running() else 0
        }


class ServiceRunner:
    """Manages multiple microservices."""
    
    def __init__(self):
        self.services: Dict[str, ServiceProcess] = {}
        self.settings = get_settings()
        self.shutdown_requested = False
    
    def configure_services(self):
        """Configure all services."""
        logger.info("Configuring services...")
        
        # Split NLP Provider Services - ports from environment or defaults
        import os
        
        # Use centralized configuration
        from src.shared.config import get_settings
        settings = get_settings()
        
        split_services = [
            ("conceptnet_provider", "ConceptNet Service", "src.services.provider_services.conceptnet_service", 
             settings.CONCEPTNET_SERVICE_PORT),
            ("gensim_provider", "Gensim Service", "src.services.provider_services.gensim_service", 
             settings.GENSIM_SERVICE_PORT),
            ("spacy_provider", "SpaCy Service", "src.services.provider_services.spacy_service", 
             settings.SPACY_SERVICE_PORT),
            ("heideltime_provider", "HeidelTime Service", "src.services.provider_services.heideltime_service", 
             settings.HEIDELTIME_SERVICE_PORT)
        ]
        
        for service_key, service_name, module_name, port in split_services:
            self.services[service_key] = ServiceProcess(
                name=service_name,
                module=module_name,
                port=port,
                env={"PORT": str(port)}
            )
        
        # LLM Provider Service  
        self.services["llm_provider"] = ServiceProcess(
            name="LLM Provider Service",
            module="src.services.provider_services.llm_provider_service",
            port=8002,
            env={"PORT": "8002"}
        )
        
        
        logger.info(f"Configured {len(self.services)} services")
    
    async def start_all(self, services: Optional[List[str]] = None) -> bool:
        """Start all or specified services."""
        target_services = services or list(self.services.keys())
        
        logger.info(f"Starting services: {', '.join(target_services)}")
        
        success_count = 0
        for service_name in target_services:
            if service_name in self.services:
                if await self.services[service_name].start():
                    success_count += 1
            else:
                logger.error(f"Unknown service: {service_name}")
        
        logger.info(f"Started {success_count}/{len(target_services)} services successfully")
        return success_count == len(target_services)
    
    async def stop_all(self, services: Optional[List[str]] = None) -> bool:
        """Stop all or specified services."""
        target_services = services or list(self.services.keys())
        
        logger.info(f"Stopping services: {', '.join(target_services)}")
        
        success_count = 0
        for service_name in target_services:
            if service_name in self.services:
                if await self.services[service_name].stop():
                    success_count += 1
            else:
                logger.error(f"Unknown service: {service_name}")
        
        logger.info(f"Stopped {success_count}/{len(target_services)} services successfully")
        return success_count == len(target_services)
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        logger.info(f"Restarting {service_name}...")
        
        service = self.services[service_name]
        await service.stop()
        await asyncio.sleep(1)  # Brief pause
        return await service.start()
    
    def get_status(self) -> Dict[str, any]:
        """Get status of all services."""
        return {
            "services": {name: service.get_status() for name, service in self.services.items()},
            "running_count": len([s for s in self.services.values() if s.is_running()]),
            "total_count": len(self.services)
        }
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_forever(self, services: Optional[List[str]] = None):
        """Run services until shutdown signal."""
        self.setup_signal_handlers()
        
        # Start services
        if not await self.start_all(services):
            logger.error("Failed to start all services, exiting")
            return
        
        logger.info("All services started. Running until shutdown signal...")
        
        try:
            # Monitor services
            while not self.shutdown_requested:
                await asyncio.sleep(5)
                
                # Check for failed services
                for name, service in self.services.items():
                    if not service.is_running():
                        logger.warning(f"Service {name} has stopped unexpectedly")
                        # Could implement auto-restart here
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            logger.info("Shutting down all services...")
            await self.stop_all()


# CLI interface
async def main():
    """Main CLI entry point."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Microservices Runner")
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "run"], 
                       help="Action to perform")
    parser.add_argument("--services", nargs="+", 
                       choices=["conceptnet_provider", "gensim_provider", "spacy_provider", "heideltime_provider", "llm_provider", "plugin_router"],
                       help="Specific services to target")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create runner
    runner = ServiceRunner()
    runner.configure_services()
    
    if args.action == "start":
        success = await runner.start_all(args.services)
        sys.exit(0 if success else 1)
    
    elif args.action == "stop":
        success = await runner.stop_all(args.services)
        sys.exit(0 if success else 1)
    
    elif args.action == "restart":
        if args.services:
            success = True
            for service in args.services:
                if not await runner.restart_service(service):
                    success = False
        else:
            await runner.stop_all()
            await asyncio.sleep(2)
            success = await runner.start_all()
        sys.exit(0 if success else 1)
    
    elif args.action == "status":
        status = runner.get_status()
        print(f"Services Status: {status['running_count']}/{status['total_count']} running")
        for name, service_status in status["services"].items():
            status_icon = "🟢" if service_status["running"] else "🔴"
            print(f"  {status_icon} {service_status['name']} (port {service_status['port']})")
            if service_status["running"]:
                print(f"    PID: {service_status['pid']}, Uptime: {service_status['uptime_seconds']:.1f}s")
    
    elif args.action == "run":
        await runner.run_forever(args.services)


if __name__ == "__main__":
    import os
    asyncio.run(main())