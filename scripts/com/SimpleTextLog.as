package com.wdcgame.utils
{
   import flash.display.Stage;
   import flash.events.KeyboardEvent;
   import flash.events.MouseEvent;
   import flash.external.ExternalInterface;
   import flash.sampler.getSize;
   import flash.system.System;
   import flash.text.TextField;
   import flash.text.TextFormat;
   import flash.utils.getTimer;
   import models.GameData;
   
   public class SimpleTextLog extends Singleton
   {
      
      public static var Close:Boolean = true;
      
      private var _txt:TextField;
      
      private var isZDown:Boolean = false;
      
      private var isXDown:Boolean = false;
      
      private var isCDown:Boolean = false;
      
      private var lastN:Number = getTimer();
      
      private var __test:String = "";
      
      public var message:Array = [];
      
      private var arr:Array = [];
      
      public function SimpleTextLog()
      {
         super();
         _txt = new TextField();
         _txt.width = 400;
         _txt.wordWrap = true;
         var _loc1_:TextFormat = new TextFormat("verdana",14,16777215);
         _txt.defaultTextFormat = _loc1_;
         _txt.height = 160;
         _txt.type = "input";
         _txt.visible = false;
         _txt.y = 300;
      }
      
      public static function getIns() : SimpleTextLog
      {
         return Singleton.getIns(SimpleTextLog);
      }
      
      public function get txt() : TextField
      {
         return _txt;
      }
      
      protected function onc(param1:MouseEvent) : void
      {
         _txt.visible = false;
         _txt.selectable = false;
         _txt.removeEventListener("mouseOut",onc);
      }
      
      protected function onMouse(param1:MouseEvent) : void
      {
         var _loc2_:Stage = param1.currentTarget as Stage;
         if(_loc2_.mouseX < 20)
         {
            _txt.visible = true;
            _txt.selectable = true;
            _txt.addEventListener("mouseOut",onc);
         }
      }
      
      public function init(param1:Stage) : void
      {
         param1.addChild(_txt);
         param1.addEventListener("keyDown",__keydown);
         param1.addEventListener("keyUp",__keyup);
      }
      
      protected function __keydown(param1:KeyboardEvent) : void
      {
         if(param1.keyCode == 90)
         {
            isZDown = true;
         }
         if(param1.keyCode == 88)
         {
            isXDown = true;
         }
         if(param1.keyCode == 67)
         {
            isCDown = true;
         }
      }
      
      protected function __keyup(param1:KeyboardEvent) : void
      {
         if(param1.keyCode == 90)
         {
            isZDown = false;
         }
         if(param1.keyCode == 88)
         {
            isXDown = false;
         }
         if(param1.keyCode == 67)
         {
            isCDown = false;
         }
         if(param1.keyCode == 80)
         {
            if(isZDown && isXDown && isCDown)
            {
               _txt.visible = !_txt.visible;
               System.setClipboard(__test);
               if(_txt.visible)
               {
                  _txt.text = __test;
               }
            }
         }
      }
      
      public function log(... rest) : void
      {
         if(_txt.parent)
         {
            _txt.parent.setChildIndex(_txt,_txt.parent.numChildren - 1);
         }
         if(_txt.maxScrollV > 1000)
         {
         }
         var _loc3_:Number = getTimer();
         var _loc2_:String = "-->" + rest + "\n";
         lastN = _loc3_;
         __test += rest + "\n";
         _txt.appendText(_loc2_);
         _txt.scrollV = _txt.maxScrollV;
         trace("log=" + rest);
         try
         {
            ExternalInterface.call("console.log",rest);
         }
         catch(error:Error)
         {
         }
      }
      
      public function log2(... rest) : void
      {
         if(Close)
         {
            return;
         }
         if(GameData.isRelease === false || GameData.uid == 682337385 || GameData.uid == 99181269)
         {
            if(!__test)
            {
               __test = rest + "\n";
            }
            else
            {
               __test += rest + "\n";
            }
            trace("" + rest);
            try
            {
               ExternalInterface.call("console.log",rest);
            }
            catch(error:Error)
            {
            }
         }
      }
      
      public function log3(... rest) : void
      {
      }
      
      public function clear() : void
      {
         _txt.text = "";
      }
      
      public function logErr(... rest) : void
      {
         if(!message)
         {
            message = [rest];
         }
         else
         {
            message.push(rest);
         }
         log(rest);
      }
      
      public function Copy() : void
      {
      }
      
      public function DebugMem(param1:*, param2:*) : void
      {
         if(GameData.uid == 100)
         {
            log(param1 + " 占kb=" + Math.floor(getSize(param2) / 1024));
         }
      }
   }
}

