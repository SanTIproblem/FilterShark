var map = new BMap.Map("bmap", {
    coordsType: 5             
 });            
 map.disableBizAuthLogo(); //关闭
 var point = new BMap.Point(112.22, 31.05);  
 var marker = new BMap.Marker(point);        // 创建标注    
 map.addOverlay(marker);           
 map.centerAndZoom(point, 13);             
 map.enableScrollWheelZoom(true);  
 map.addControl(new BMap.NavigationControl());      
map.addControl(new BMap.ScaleControl());    
map.addControl(new BMap.OverviewMapControl());    
map.addControl(new BMap.MapTypeControl());   
function get() {
    map.clearOverlays();   
    var X= document.getElementById('searchtext').value;
    var GG = new BMap.Geocoder();
    GG.getPoint(X, function (point) {
        if (point) {
            map.centerAndZoom(point, 15);   
            map.addOverlay(new BMap.Marker(point, { title: X }))    
        } else {
            alert('您选择的地址没有解析到结果！');
        }
    }, "")
}