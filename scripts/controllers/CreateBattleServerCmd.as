package controllers
{
   import com.wdcgame.engine.core.GameFrame;
   import com.wdcgame.engine.model.LevelConfigVO;
   import com.wdcgame.frame.ApplicationFacade;
   import com.wdcgame.utils.SimpleTextLog;
   import com.ywxgame.models.manager.BagModelManager;
   import com.ywxgame.models.manager.EventManager;
   import com.ywxgame.models.manager.EveryDaySetting;
   import flash.display.DisplayObject;
   import models.GameData;
   import models.managers.BattleModelManager;
   import models.managers.CommandModelManager;
   import models.managers.TowerModelManager;
   import models.managers.UserModelManager;
   import models.vo.BattleLevelParam;
   import models.vo.RequestVO;
   import modules.utils.CreateComponent;
   import org.puremvc.as3.interfaces.IMediator;
   import org.puremvc.as3.interfaces.INotification;
   import org.puremvc.as3.patterns.command.SimpleCommand;
   
   public class CreateBattleServerCmd extends SimpleCommand
   {
      
      private var _levelParam:BattleLevelParam;
      
      public function CreateBattleServerCmd()
      {
         super();
      }
      
      override public function execute(param1:INotification) : void
      {
         var rout:String;
         var param:Object;
         var req:RequestVO;
         var previewCfg:Object;
         var notification:INotification = param1;
         "cmdCreateBattleServer";
         _levelParam = notification.getBody() as BattleLevelParam;
         if(!_levelParam)
         {
            throw new Error("CreateBattleCmd.body参数不对");
         }
         SimpleTextLog.getIns().log("CreateBattleServerCmd_levelParam.Config.id=" + _levelParam.Config.id);
         previewCfg = _levelParam.Config;
         SimpleTextLog.getIns().log("===== 战斗前预览 =====");
         if(previewCfg != null)
         {
            SimpleTextLog.getIns().log("关卡: " + String(previewCfg["name"]) + " (ID=" + String(previewCfg["id"]) + ")");
            SimpleTextLog.getIns().log("类型: " + String(previewCfg["type"]));
            SimpleTextLog.getIns().log("难度: " + String(previewCfg["difficult"]));
         }
         if(_levelParam.isPetPk)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
            return;
         }
         if(_levelParam.isUnionBoss)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
            return;
         }
         if(_levelParam.isWorldBoss)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
            return;
         }
         if(_levelParam.isPvPRobot)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
            return;
         }
         if(_levelParam.isPvPVideo)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
            return;
         }
         if(_levelParam.isVideo)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
            return;
         }
         if(_levelParam.Config.countdownMode > 0)
         {
            rout = "scene.demonTowerHandler.beginChallenge";
            param = {"type":TowerModelManager.getIns().type};
            req = new RequestVO(rout,param,null,function(param1:Object):void
            {
               var _loc2_:DisplayObject = null;
               if(param1.code == 200)
               {
                  facade.sendNotification("CMD_CONSUME_ITEM",param1);
                  facade.sendNotification("cmdCreateBattle",_levelParam);
               }
               else
               {
                  ApplicationFacade.getInstance().sendNotification("nfLoadingAssetscomplete");
                  CommandModelManager.gtoToLastTown();
                  _loc2_ = CreateComponent.createTopAutoWarnBar(param1.error,6);
                  if(_loc2_ && GameData.stage)
                  {
                     GameData.stage.addChild(_loc2_);
                  }
               }
            });
            facade.sendNotification("cmdRequestServer",req);
            return;
         }
         if(_levelParam.Config.isNewSkyTree())
         {
            rout = "scene.copyHandler.beginDungeonWing";
            param = {};
            req = new RequestVO(rout,param,null,function(param1:Object):void
            {
               var _loc2_:DisplayObject = null;
               if(param1.code == 200)
               {
                  _levelParam.drops = param1.drop;
                  facade.sendNotification("cmdCreateBattle",_levelParam);
               }
               else
               {
                  ApplicationFacade.getInstance().sendNotification("nfLoadingAssetscomplete");
                  CommandModelManager.gtoToLastTown();
                  _loc2_ = CreateComponent.createTopAutoWarnBar(param1.error,6);
                  if(_loc2_ && GameData.stage)
                  {
                     GameData.stage.addChild(_loc2_);
                  }
               }
            });
            facade.sendNotification("cmdRequestServer",req);
            return;
         }
         if(_levelParam.Config.isChapter())
         {
            rout = "scene.chapterRoleHandler.beginChapterDungeon";
            param = {"dungeonId":_levelParam.Config.id};
            req = new RequestVO(rout,param,null,function(param1:Object):void
            {
               if(param1.code == 200)
               {
                  facade.sendNotification("cmdCreateBattle",_levelParam);
               }
               else
               {
                  ApplicationFacade.getInstance().sendNotification("nfLoadingAssetscomplete");
                  CommandModelManager.gtoToLastTown();
               }
            });
            facade.sendNotification("cmdRequestServer",req);
            return;
         }
         if(_levelParam.isCard == false)
         {
            if(_levelParam.isMultiplayer == false)
            {
               rout = "scene.copyHandler.createSinglePlayerCopy";
               param = {
                  "mid":_levelParam.Config.id,
                  "costGem":_levelParam.CostGem,
                  "rnd":_levelParam.MyRand
               };
               req = new RequestVO(rout,param,null,onC);
               req.isErrorBack = false;
               facade.sendNotification("cmdRequestServer",req);
            }
            else
            {
               facade.sendNotification("cmdLoginServerTeam",_levelParam.Response);
            }
         }
         else if(_levelParam.isCard && _levelParam.isPreview == false)
         {
            rout = "scene.cardHandler.beginBattle";
            if(_levelParam.cardUserData.type == 2)
            {
               rout = "scene.cardHandler.beginBattleWithTicket";
            }
            param = {"entityId":_levelParam.cardUserData.entityId};
            req = new RequestVO(rout,param,null,onCardBack);
            facade.sendNotification("cmdRequestServer",req);
         }
         else if(_levelParam.isCard && _levelParam.isPreview)
         {
            facade.sendNotification("cmdCreateBattle",_levelParam);
         }
         if(GameFrame.stage)
         {
            GameFrame.stage.focus = null;
         }
      }
      
      private function onCardBack(param1:Object) : void
      {
         if(param1.code != 200)
         {
            facade.sendNotification("cmdGoToTown",UserModelManager.getIns().getLastEnterTown());
            facade.sendNotification("nfLoadingAssetscomplete");
            return;
         }
         UserModelManager.getIns().getCurretPlayerInfo().UpdateEveryDaySetting(param1);
         EveryDaySetting.getIns().initData(param1);
         this.handResponse(param1);
         facade.sendNotification("CMD_GIVE_ITEM",param1);
         facade.sendNotification("UPDATE_ITEM");
         facade.sendNotification("cmdCreateBattle",_levelParam);
      }
      
      private function onC(param1:Object) : void
      {
         var _loc3_:DisplayObject = null;
         var _loc2_:IMediator = null;
         if(param1.code != 200)
         {
            facade.sendNotification("cmdGoToTown",UserModelManager.getIns().getLastEnterTown());
            _loc3_ = CreateComponent.createTopAutoWarnBar(param1.error,6);
            if(_loc3_)
            {
               _loc2_ = facade.retrieveMediator("loadingLayer");
               _loc2_.getViewComponent().addChild(_loc3_);
            }
            facade.sendNotification("nfLoadingAssetscomplete");
            EventManager.emit("event_enter_battle_error");
            return;
         }
         _levelParam.ServerRand = param1.rnd;
         if(_levelParam.Config.isMutilGm())
         {
            randMonster(param1);
         }
         this.handResponse(param1);
         facade.sendNotification("cmdCreateBattle",_levelParam);
         _levelParam = null;
      }
      
      private function handResponse(param1:Object) : void
      {
         var _loc3_:int = 0;
         var _loc2_:int = -1;
         if(param1.hasOwnProperty("count"))
         {
            _loc2_ = int(param1.count);
         }
         if(param1.hasOwnProperty("itemId") && param1.hasOwnProperty("count"))
         {
            _loc3_ = int(param1.itemId);
            if(BagModelManager.getIns().getIsConsumeItemById(_loc3_))
            {
               BagModelManager.getIns().type = "item use";
            }
            else
            {
               BagModelManager.getIns().type = "item common";
            }
            BagModelManager.getIns().removeItemById(_loc3_,_loc2_);
         }
         else if(param1.hasOwnProperty("itemId") == false && param1.hasOwnProperty("count"))
         {
            UserModelManager.getIns().getCurretPlayerInfo().collectPower = _loc2_;
         }
         else if(param1.hasOwnProperty("consume"))
         {
            facade.sendNotification("CMD_CONSUME_ITEM",param1);
         }
         var debugDrops:Object = param1.drops;
         var debugCount:int = 0;
         SimpleTextLog.getIns().clear();
         SimpleTextLog.getIns().log("===== 掉落数据 buildDrops =====");
         if(debugDrops == null)
         {
            SimpleTextLog.getIns().log("drops为空");
         }
         else
         {
            SimpleTextLog.getIns().log("param1类型: " + typeof debugDrops);
            for(var debugKey in debugDrops)
            {
               var debugItem:Object = debugDrops[debugKey];
               if(debugItem == null)
               {
                  var debugLine:String = "null";
               }
               else
               {
                  debugLine = "Object {" + "index=" + debugItem.index + ", " + "count=" + debugItem.count + ", " + "type=" + debugItem.type + ", " + "subType=" + debugItem.subType + ", " + "monsterId=" + debugItem.monsterId + ", " + "entityId=" + debugItem.entityId + "}";
               }
               SimpleTextLog.getIns().log("key[" + debugCount + "] = " + debugLine);
               debugCount++;
            }
            SimpleTextLog.getIns().log("总key数: " + debugCount);
         }
         BattleModelManager.getIns().buildDrops(param1.drops);
         BattleModelManager.getIns().buildMonsters(param1.selectMonster);
         if(_levelParam.isPK)
         {
         }
      }
      
      private function onC2(param1:Object) : void
      {
         param1;
      }
      
      private function randMonster(param1:Object) : void
      {
         var _loc9_:int = 0;
         var _loc7_:int = 0;
         var _loc4_:Object = null;
         var _loc8_:int = 0;
         var _loc5_:Array = [];
         var _loc6_:Array = param1.drops || [];
         var _loc3_:Array = [];
         _loc9_ = 0;
         while(_loc9_ < _loc6_.length)
         {
            if(_loc3_.indexOf(Number(_loc6_[_loc9_].monsterId)) == -1)
            {
               _loc3_.push(Number(_loc6_[_loc9_].monsterId));
            }
            _loc9_++;
         }
         var _loc2_:LevelConfigVO = _levelParam.Config;
         _levelParam.clearRandomMonsterBirth();
         _loc7_ = 0;
         while(_loc7_ < _loc2_.preloadMonsterRes.length)
         {
            _loc4_ = _loc2_.preloadMonsterRes[_loc7_];
            if(_loc4_.hasOwnProperty("mRandom"))
            {
               _loc8_ = 0;
               while(_loc8_ < _loc4_.mIds.length)
               {
                  if(_loc3_.indexOf(_loc4_.mIds[_loc8_]) > -1 || _loc4_.mIds.length == 1)
                  {
                     _levelParam.addRandomMonsterBirth(_loc4_.birthId,_loc4_.mIds[_loc8_]);
                     _loc5_.push(_loc4_.mIds[_loc8_]);
                     break;
                  }
                  _loc8_++;
               }
            }
            _loc7_++;
         }
      }
   }
}

