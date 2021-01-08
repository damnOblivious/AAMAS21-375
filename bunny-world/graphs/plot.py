import matplotlib.pyplot as plt

def runningAvg(mylist, windowSize = 10):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(mylist, 1):

        cumsum.append(cumsum[i-1] + x)
        if i < windowSize:
            moving_ave = cumsum[i] / (i)
        else:
            moving_ave = (cumsum[i] - cumsum[i-windowSize])/windowSize
        moving_aves.append(moving_ave)

    return moving_aves

def averageVector(vectors):
    vecNum = len(vectors)
    average = []
    min = len(vectors[0])
    for vec in vectors:
        if min > len(vec): min = len(vec)

    for i in range(min):
        sum = .0
        for j in range(vecNum):
            sum += vectors[j][i]
        average.append( sum / vecNum )

    return average

"""
    by default
        epsilon = .999
        alpha = .0001
"""

"""
    sim: simultaneous DDPG
"""


alpha0005_1 = [-40.1, -173.2, 1091.3, 1309.2, 865.5, 965.7, 1071.25, 956.95, 1365.5, 1405.1, 1157.15, 1233.05, 1417.7, 1181.4, 1414.6, 1037.85, 1394.45, 1342.25, 1195.15, 1411.8, 1156.45, 1416.85, 1365.6, 1448.45, 1464.6, 1362.65, 1308.2, 1409.95, 1308.85, 1465.35, 1023.8, 1263.4, 1309.55, 1299.15, 1159.85, 1260.0, 776.85, 1375.75, 1200.15, 1206.4, 911.25, 1086.25, 907.45, 705.8, 1216.15, 1104.95, 1303.2, 1086.55, 1320.1, 1150.1, 1098.8, 1223.15, 1416.45, 1155.35, 995.55, 1093.95, 1357.35, 1368.0, 1259.95, 1414.0, 1311.3, 1307.3, 1316.15, 1352.8, 1412.75, 1459.3, 1108.75, 1365.05, 1007.7, 1210.95, 1158.3, 1366.65, 1203.6, 1313.4, 1367.45, 1310.35, 1114.4, 1158.55, 1362.45, 615.05, 1250.6, 1463.85]

alpha0005_2 = [1107.75, -183.95, 544.15, 1268.6, 1235.9, 1083.85, 1002.15, 1018.75, 1211.9, 527.45, 949.7, 752.35, 1197.0, 1018.45, 867.05, 897.7, 925.0, 1239.8, 857.05, 1153.65, 1335.4, 1243.55, 1462.8, 920.9, 1421.25, 1318.85, 1195.0, 1336.5, 1057.2, 1345.1, 1142.65, 1122.7, 852.55, 665.65, 1404.6, 1258.85, 1370.85, 1197.1, 1155.6, 1017.65, 1460.9, 1120.1, 1309.75, 1170.45, 906.35, 999.05, 1210.4, 977.7, 1055.8, 1086.65, 750.9, 474.65, 814.85, 720.8, 916.45, 838.45, 1128.3, 1145.9, 928.6, 1015.5, 1299.35, 589.25, 882.35, 1300.4, 865.2, 1360.55, 1305.05, 1034.05, 1146.15, 1165.5, 989.65, 1410.75, 1168.6, 1039.15, 986.3, 1369.2, 1153.25, 1046.25, 1255.05, 1074.85]

alpha0005_3 = [-150.45, 1098.25, 729.65, 1125.4, 921.55, 617.95, 724.55, 456.65, 1147.35, 459.5, 603.8, 1143.05, 907.55, 1041.2, 1077.0, 1095.6, 1057.9, 977.85, 1258.85, 1098.5, 1172.75, 1035.2, 847.3, 1312.7, 1201.6, 1345.1, 1078.35, 1243.0, 684.1, 1220.0, 945.2, 1366.35, 1102.6, 967.55, 1259.8, 952.4, 1091.4, 1000.45, 1321.8, 665.75, 900.45, 1388.65, 1269.1, 1384.1, 1392.7, 571.1, 1350.1, 1308.85, 1462.65, 1357.45, 1464.7, 1358.2, 1065.95, 1246.45, 1229.95, 1352.0, 1125.25, 1353.75, 762.65, 911.65, 1406.9, 1094.1, 1285.65, 1100.55, 812.1, 332.25, 1237.65, 1205.85, 394.45, 1290.35, 1025.6, 1296.3, 1395.25, 723.7, 1311.5, 678.3, 926.8, 55.35, 873.35, 1030.9, 1273.55, 1211.7]

alpha0005_4 = [-98.65, -203.2, 1005.65, 1251.5, 990.65, 1227.55, 1263.95, 1319.25, 1307.4, 1152.45, 1151.95, 1365.25, 1141.15, 1312.2, 1093.1, 1348.05, 1361.85, 1342.0, 1469.4, 1143.9, -38.05, 1187.4, 1342.15, 1328.55, 1274.2, 1302.05, 1357.4, 1383.8, 1355.4, 1334.15, 1253.75, 1106.9, 557.65, 1261.4, 1335.6, 1241.95, 1381.25, 1327.5, 1312.8, 1096.0, 1361.5, 975.2, 1120.55, 1126.95, 973.65, 1288.35, 455.1, 1454.75, 1028.1, 1281.75, 1275.4, 1379.55, 1183.3, 800.15, 995.85, 1102.2, 1007.25, 589.15, 775.0, 1254.45, 663.0, 1040.75, 1004.95, 838.1, 617.6, 987.8, 949.8, 800.75, 888.7, 1227.0, 1364.8, 695.15, 1059.0, 1006.5, 1383.5, 1312.15, 1336.0, 842.1, 683.35, 1109.35, 1097.4, 1163.55, 1235.55, 868.6, 1234.35, 1153.2, 1363.0, 1274.1, 1079.45, 1420.1, 1223.3]

alpha0002_1 = [-120.4, -191.1, -228.85, -157.3, 32.1, 280.25, 634.4, 1463.65, 1263.5, 1353.2, 999.4, 1391.2, 655.8, 1227.85, 817.9, 1241.8, 1060.0, 1305.05, 1273.05, 1231.8, 1409.9, 966.8, 1304.5, 1191.5, 1246.25, 1414.2, 1329.0, 1288.15, 1305.4, 1398.4, 1407.05, 1204.65, 1469.15, 1260.65, 1413.6, 1190.85, 1328.5, 1084.75, 1248.25, 1064.2, 1185.25, 1148.7, 1386.15, 1344.6, 1352.6, 1315.85, 1282.15, 1469.35, 1273.95, 1264.65, 1354.45, 1363.85, 1361.75, 1304.8, 1358.55, 1335.2, 1249.85, 1418.05, 1235.75, 1358.85, 1467.45, 1304.65, 1355.4, 1404.35, 1418.5, 1370.35, 1417.2, 1244.25, 1463.95, 961.8, 994.35, 1463.65, 795.2, 1271.1, 1348.65, 1458.55, 1413.75, 1303.05, 1216.7, 1276.4, 1000.85, 1381.2, 1224.6, 1357.4, 1185.15, 1226.65, 1217.8, 1386.35, 1397.35, 971.55, 1177.95, 1369.5, 1302.2, 1454.55, 1336.05, 1089.1, 1455.35]

alpha0002_2 = [514.45, 134.15, 992.65, -121.25, 210.65, 1092.2, 1216.45, 1258.3, 1020.8, 1200.7, 1167.7, 1330.5, 1318.25, 695.8, 1212.3, 1308.55, 911.5, 1166.85, 1407.2, 1364.75, 1355.35, 1103.55, 1390.45, 1417.55, 1317.4, 1354.0, 1418.5, 1366.55, 1467.5, 1307.05, 1200.8, 1464.9, 1422.6, 1268.6, 1339.45, 1228.9, 1365.15, 1250.85, 1414.8, 1359.65, 1304.8, 1147.6, 1200.7, 1208.35, 1365.95, 1211.7, 1241.85, 194.3, 935.6, 1032.1, 1299.2, 1290.2, 1086.65, 1122.1, 1220.5, 1173.75, 1074.4, 1178.25, 1016.7, 1295.95, 1303.0, 1470.4, 1469.4, 1259.25, 1357.75, 1413.9, 1299.6, 1338.45, 1472.1, 1314.15, 1415.0, 1415.15, 1148.8, 1418.45, 1411.75, 1337.4, 1248.05, 1308.45, 1473.25, 1463.25, 1471.5, 1463.5, 1466.05, 1416.5, 1365.8, 1349.8, 1349.1, 1232.65, 1292.65, 990.6, 1356.25, 1327.1, 1360.6, 1371.45, 1470.95, 1347.25, 1416.65, 1349.75, 1376.6, 1405.7, 1308.9, 1472.05, 1463.1, 1334.0]

alpha0002_3 = [-147.2, -170.75, -165.9, -129.15, -214.05, -159.75, -171.1, -205.35, -211.55, -192.45, -145.45, -103.75, 120.9, 439.3, 826.55, 831.2, 553.4, 954.35, 1258.1, 1210.15, 865.45, 1229.45, 953.9, 1102.25, 1312.95, 1194.25, 1390.15, 1331.8, 1044.9, 1255.25, 1224.85, 1083.6, 1048.3, 1112.55, 1312.0, 1471.0, 1386.75, 1265.4, 1139.9, 1410.05, 1397.55, 1306.85, 1185.5, 1157.85, 1337.7, 1197.95, 906.75, 1340.15, 1320.1, 1126.45, 1336.4, 1307.75, 1120.0, 649.0, 811.5, 1410.2, 1367.05, 1343.95, 1042.55, 1000.8, 1153.5, 1402.2, 1277.0, 1286.9, 1239.7, 1361.9, 1293.4, 1303.2, 1036.0, 1290.85, 858.25, 882.45, 1109.7, 1132.6, 1275.65, 1218.0, 1360.55, 1407.3, 1417.9, 1405.7, 1414.35, 1391.7, 1466.6, 1410.1, 1421.05, 1395.4, 1238.65, 1335.55, 1211.0, 1462.75, 1295.9, 1441.25, 1254.1, 1350.75, 1327.3, 1295.15, 1392.6, 1470.0, 1462.5, 1332.15, 1356.8, 1233.0, 1468.4, 1416.8]

alpha0002_4 = [1422.7, 946.5, 742.2, 1418.95, 1076.5, 1096.3, 1099.35, 1363.5, 1420.35, 1418.85, 1317.2, 1203.95, 1462.35, 1416.2, 1136.35, 1377.85, 1076.2, 1222.15, 1274.75, 1466.1, 1411.65, 1417.5, 1044.45, 1473.4, 1159.7, 1468.65, 1354.65, 1282.6, 1290.95, 1301.9, 1263.45, 1290.9, 1326.3, 1069.85, 1142.95, 1311.75, 840.5, 1362.25, 1346.25, 1195.55, 1109.0, 1416.75, 1125.3, 1413.2, 1314.75, 937.8, 1263.05, 859.9, 1259.95, 1341.25, 1128.1, 1422.75, 1164.9, 839.2, 742.95, 710.7, 514.75, 897.4, 1195.45, 1411.65, 1353.45, 1361.75, 1411.95, 1302.3, 1315.3, 1452.2, 1463.85, 1332.65, 1188.45, 1278.35, 1406.9, 1467.4, 1357.65, 1316.9, 1411.95, 1423.75, 1418.95, 1281.2, 1414.0, 1362.0, 1070.3, 1360.45, 1119.95, 1183.05, 1166.85, 1402.1, 1338.45, 1307.9, 1470.15, 1217.8, 1286.55, 1361.45, 1142.1, 1099.6, 1137.05, 1081.5, 1348.25, 1332.8, 1053.85, 825.0, 1142.0, 1242.8, 979.65, 848.35]

alpha0001_1 = [-125.75, -177.8, -135.1, 1412.5, 1423.25, 1463.0, 1360.45, 1055.2, 1103.9, 960.9, 1080.65, 1409.5, 1421.4, 1419.4, 1457.3, 1418.8, 1472.4, 1469.7, 1294.45, 1108.9, 1280.5, 968.25, 1297.1, 1044.2, 1118.1, 846.8, 1110.7, 1118.75, 929.2, 1208.25, 1239.85, 924.45, 1077.45, 1468.6, 1325.95, 1351.8, 1183.2, 1463.4, 1469.3, 1114.55, 1473.05, 1288.45, 1421.2, 1465.1, 1232.35, 1360.05, 1210.8, 1353.35, 1465.35, 1273.2, 1322.1, 1348.85, 1163.85, 1416.3, 1418.35, 1249.4, 1261.0, 1148.35, 1304.7, 1418.3, 1311.5, 1276.9, 1130.85, 1467.6, 1474.9, 1417.15, 1467.55, 1239.55, 1383.6, 1459.6, 1472.8, 1468.0, 1367.1, 1413.4, 1182.75, 1471.65, 1422.65, 1336.95, 1400.95, 1317.2, 1363.65, 1333.1, 1470.5, 1369.4, 1335.55, 1387.4, 1048.15, 1363.35, 1415.35, 1418.7, 1470.9, 1084.5, 1401.75, 1364.5, 1470.35, 1236.5, 1255.1, 1348.0, 1403.45, 1267.55, 1402.15, 1262.35, 1458.5, 1372.15]

alpha0001_2 = [-436.35, 934.8, 1260.9, -30.8, 109.05, 355.7, 808.6, 937.5, 712.95, 423.85, 576.05, 430.9, 371.5, 110.4, 136.35, 246.75, 61.0, 96.9, -25.3, 240.7, 1001.15, 1083.1, 825.4, 961.3, 1151.5, 1229.95, 1315.3, 1277.4, 911.4, 1413.6, 1397.25, 1009.85, 1147.4, 1462.05, 1474.1, 1464.8, 1061.05, 1343.4, 1377.25, 1408.95, 831.7, 1234.3, 1122.2, 1371.95, 1307.75, 1249.75, 1173.65, 1301.3, 1413.85, 1343.75, 1458.55, 790.3, 830.95, 871.5, 1404.3, 1235.4, 1154.55, 1303.1, 1403.95, 1319.6, 1011.35, 1403.95, 1020.85, 1403.8, 1326.95, 1245.55, 978.55, 1465.95, 1230.2, 1399.45, 1240.95, 1269.4, 1221.5, 1285.85, 1275.8, 1332.2, 1347.8, 1419.25, 1127.25, 1360.25, 1199.9, 1336.3, 1366.0, 1314.2, 1410.45, 1305.65, 1112.45, 1346.55, 1294.55, 1417.55, 1351.5, 1096.6, 1092.25, 1215.8, 1246.0, 1098.45, 1304.75, 1237.9, 1353.75, 1134.45, 1185.0, 1057.45, 886.4, 1248.1]

alpha0001_3 = [-207.6, -182.9, -229.15, -38.45, 861.3, 321.2, -176.75, -43.85, 334.25, 276.25, 1358.85, 1289.65, 1165.7, 1268.55, 1216.5, 1168.35, 1215.1, 1194.7, 1257.95, 1136.35, 1043.55, 1246.9, 1166.2, 1051.7, 1216.6, 1231.65, 1241.15, 1416.7, 1207.6, 1231.25, 901.65, 1223.25, 1079.25, 1388.25, 1181.85, 1187.9, 977.85, 1111.15, 1394.35, 1044.35, 1099.9, 1313.45, 1289.1, 1255.6, 1353.6, 1362.55, 1186.05, 1182.75, 1311.8, 1368.45, 1369.3, 1391.95, 1415.45, 1356.45, 1418.8, 1255.6, 1365.75, 1472.5, 1332.8, 1177.75, 1391.0, 1473.35, 1305.7, 1287.35, 1362.5, 1236.0, 1417.9, 1227.7, 1306.6, 1467.75, 1296.7, 1168.6, 1402.6, 1187.35, 1364.6, 1368.45, 1423.9, 1202.35, 1210.2, 1333.6, 1162.5, 1358.6, 1198.2, 1464.5, 1348.95, 1199.15, 1265.15, 1355.55, 1418.15, 1122.45, 1187.95, 1412.5, 1422.65, 1146.0, 1302.05, 1357.55, 1362.8, 1464.35, 1248.1, 1413.6, 1205.35, 1229.8, 1322.6, 1418.9]

alpha0001_4 = [-157.25, -281.8, -205.8, -234.05, 446.2, 765.55, 1163.15, 991.4, 627.85, 1018.65, 852.55, 968.7, 747.75, 1381.7, 960.25, 1416.7, 1351.2, 1248.7, 1299.3, 1198.95, 912.25, 945.1, 1003.1, 812.75, 747.5, 1164.25, 1280.15, 1173.15, 1108.9, 1044.7, 1187.95, 1055.9, 832.85, 1258.6, 1413.7, 1144.95, 1362.45, 1263.5, 1207.25, 1120.45, 1133.6, 1136.75, 1106.6, 1156.3, 1266.3, 535.95, 1112.25, 1304.8, 1315.1, 1149.6, 1165.0, 1109.1, 1164.95, 1360.6, 1455.95, 1205.5, 1302.85, 1420.1, 1331.4, 1313.2, 1354.9, 1321.25, 1300.9, 1153.1, 867.65, 1142.1, 1313.1, 909.15, 1363.3, 1246.1, 1313.4, 1218.0, 1304.3, 1170.9, 1315.8, 1151.5, 1240.15, 1422.5, 1266.0, 950.85, 1110.1, 1201.6, 1307.8, 1372.55, 1340.7, 1118.45, 1126.55, 1304.9, 1002.65, 1039.75, 1349.0, 899.6, 1103.9, 1143.55, 1130.95, 1157.95, 1073.7, 1356.75, 1055.4, 1236.35, 1206.65, 1021.3, 1225.85, 1315.75]

alpha00002_1 = [-248.05, -182.15, -152.85, -175.75, -229.55, -215.15, -185.3, -256.3, -190.85, -163.75, 1460.75, -122.15, -163.4, -208.85, -284.45, -74.7, -210.65, -187.9, -134.05, -230.45, -175.75, -231.35, -189.05, -214.9, -239.45, -141.7, -207.1, -247.55, -147.15, -165.55, -153.5, -223.25, -201.25, -181.65, -156.1, -93.5, -125.9, -212.5, -167.65, -123.1, -100.15, -204.1, -150.35, -61.85, -166.9, 0.95, -61.7, 750.75, 701.15, 403.9, 428.55, 812.55, 998.3, -47.7, 914.45, 736.65, 741.7, 691.0, 1030.15, 704.05, 665.85, 1080.1, 997.65, 775.65, 1081.3, 1006.2, 853.15, 789.1, 1018.0, 1015.65, 1262.1, 877.05, 1089.15, 1055.0, 1248.25, 1141.4, 1357.55, 1236.95, 1211.0, 1227.0, 1309.7, 1318.95, 1337.35, 1286.95]

alpha00002_2 = [-189.7, -191.1, -206.9, -184.55, -194.45, -192.1, -171.25, -105.3, -155.0, -179.85, -155.05, -148.45, -149.55, -159.9, -132.8, -159.1, -204.1, -254.25, -117.7, -187.05, -199.4, -124.45, -209.05, -123.85, -137.55, -248.7, -212.45, -67.45, -177.9, -231.95, -218.75, -193.6, -204.25, -85.55, -55.35, -151.25, -187.25, -180.75, -104.0, -219.3, -205.9, -243.1, -209.2, -252.85, 394.9, 266.35, 301.5, 544.7, 368.7, -186.3, -177.95, -216.3, -158.0, -186.15, -195.35, -166.2, -165.85, -172.95, -96.95, -135.4, -234.75, -241.5, -139.35, -249.1, -216.65, -124.9, -237.85, -154.0, -194.45, -202.55, -226.25, -257.85, -174.9, -159.1, -157.75, -261.95, -182.95, -209.25, -183.85, -211.1, -124.45, -169.05, -174.8, -158.9, -187.05, -168.1, -151.0, -241.7, -228.2, -254.95, -199.9, -271.85, -159.7, -179.15, -200.45, -181.75, -239.7, -208.1, -250.25, -133.0, -189.95, -230.65, -197.15, -171.95]

alpha00002_3 = [-200.7, -170.15, -139.15, -204.1, -129.9, -285.8, -94.35, -196.4, -221.65, -187.25, -176.35, -319.7, -253.85, -211.3, -169.4, -238.6, -256.55, -215.55, -149.5, -195.15, -221.9, -193.45, -223.6, -209.85, -216.3, -132.1, -212.25, -201.85, -159.95, -152.85, -152.85, -199.9, -151.0, -218.3, -174.4, -150.25, -164.15, -140.2, -253.85, -120.0, -211.55, -285.35, -120.0, -173.5, -177.35, -211.6, -217.15, -184.05, -153.9, -247.55, -220.25, -209.5, -231.4, -119.6, -194.4, -232.0, -105.85, -139.0, -124.05, -148.75, -124.7, -252.55, -191.05, -170.3, -259.95, -223.95, -212.1, -140.15, -98.7, -130.05, -195.4, -31.95, -46.2, -102.6, -61.0, -27.2, 44.9, 88.05, 178.3, 320.75, 331.15, 1126.25, 879.55, 701.45, 1079.25, 1038.8, 746.8, 1197.15, 1190.6, -293.15, -304.05, 597.8, 579.65, 152.6, 717.05, 756.2, 549.7, 806.85, 490.1, 303.8, 834.9, 716.7, 715.85, 703.25]

alpha00002_4 = [-125.95, -105.45, -184.65, 1006.55, 1132.05, 1255.25, 847.8, -171.2, -173.75, -208.75, -183.85, -211.1, -222.3, -223.0, -228.3, -230.35, -67.1, -225.05, -220.35, -238.1, -237.2, -205.45, -174.2, -202.95, -156.95, -201.4, -228.35, -154.15, -208.85, -196.9, -151.8, -207.5, -212.65, -65.8, -242.15, -215.15, 37.1, 195.85, 443.85, 471.3, 308.85, 205.2, 373.8, 395.55, 370.45, 418.35, 376.4, 552.9, 490.15, 414.7, 688.95, 265.65, 36.95, 102.5, 193.3, 184.1, -93.15, -82.35, -48.0, 199.85, -81.3, 86.8, -64.15, 16.35, 224.8, -106.0, 19.75, -182.55, -93.7, -69.8, -102.6, -92.25, -82.45, 18.15, 32.15, 29.05, 229.45, 68.4, 109.8, 9.35, -35.9, -14.3, -160.15, -122.3, -213.4, -85.95, -51.05, -131.7, 120.3, 401.4, 50.4, -121.5, 277.1, 256.55, 181.1, 382.35, 400.5, 441.1, 172.4, 299.9, 414.45, 623.0, 518.05, 643.9]

alpha00005_1 = [-264.4, -217.4, -230.65, -129.65, -121.05, -149.65, -167.6, 991.85, 1011.1, -198.05, -193.4, -204.85, -105.95, 327.85, 421.35, 579.3, 665.6, 737.9, 1035.05, 1468.1, 1261.6, 1342.9, 1316.3, 1156.3, 1042.95, 1094.65, 1201.1, 1357.3, 1289.05, 1473.6, 1313.6, 1215.1, 1346.05, 1321.25, 183.1, 545.4, 1354.35, -362.45, 385.45, 837.85, 822.35, 1327.4, 1358.05, 1332.75, 1251.15, 1467.45, 1463.35, 1175.4, 1361.7, 1412.75, 1305.25, 1339.45, 1352.3, 1108.9, 1150.7, 999.9, 753.6, 1099.3, 504.5, 1112.5, 1331.95, 783.1, 1471.65, 1162.6, 924.4, 473.8, 1310.45, 1330.0, 1352.8, 1469.35, 1166.75, 1417.6, 1469.75, 1234.75, 1311.55, 1240.7, 1372.6, 1314.75, 1212.6, 1389.15, 1019.6, 1191.1, 864.95, 1105.25, 1235.6, 1245.6, 1021.45, 872.35, 1048.8, 706.95, 990.6, 1185.3, 1420.05, 1161.8, 1231.15, 1144.95, 1277.5, 1214.0, 1386.7, 1303.85, 1369.5, 1365.75, 1281.9, 1113.2]

alpha00005_2 = [-86.55, -206.45, -176.9, -149.8, -135.1, -223.6, -151.8, -138.95, 8.65, -181.8, -60.2, -200.95, -170.7, -175.95, -191.25, -11.5, -121.15, 133.75, -76.85, 723.0, 1349.65, 1389.6, 1289.7, 1312.55, 1394.45, 1467.25, 1466.7, 1414.2, 1466.25, 1381.45, 1412.7, 1461.5, 1408.8, 1381.45, 1379.7, 1397.35, 1407.15, 1411.25, 1417.35, 1187.35, 1263.7, 1208.85, 1402.65, 1471.5, 1182.7, 1344.7, 1384.15, 865.7, 723.9, 1253.05, 905.4, 624.0, 961.4, 906.7, 905.9, 928.7, 1000.3, 1045.6, 1346.85, 1121.9, 965.55, 1205.55, 1191.75, 978.55, 1153.05, 1249.0, 1152.5, 1283.35, 1243.35, 1189.15, 1470.1, 1152.3, 1363.55, 1368.65, 1109.3, 1330.2, 1078.2, 1349.05, 1228.3, 1368.9, 1204.85, 1167.0, 1406.5, 1313.9, 1304.2, 1305.5, 1372.25, 1258.3, 1414.75, 1353.85, 1409.4, 1294.4, 1407.35, 1468.15, 1357.35, 1226.3, 1350.9]

alpha00005_3 = [684.6, 138.15, 363.25, 426.45, 24.05, 398.05, 507.2, 225.5, 318.55, 1013.2, 251.95, 346.75, 33.25, 337.1, 864.45, 1267.25, 1036.45, 1067.8, 1222.7, 1214.1, 1192.1, 1031.6, 1099.75, 1239.4, 927.95, 711.1, 1278.75, 1100.85, 1265.95, 1348.7, 1336.2, 1315.05, 1414.4, 1174.2, 1232.5, 1140.0, 1202.5, 1364.95, 1256.75, 1417.6, 1340.8, 1237.1, 1328.05, 1466.9, 1360.2, 1468.1, 1110.4, 1216.55, 1371.0, 1378.1, 1259.9, 1189.5, 982.9, 986.7, 883.5, 882.7, 850.85, 1103.35, 633.6, 609.35, 930.25, 661.3, 624.3, 966.95, 829.45, 735.3, 688.5, 601.5, 1240.5, 1172.25, 1091.35, 1373.6, 1417.35, 1255.05, 1470.4, 1259.5, 1460.6, 1195.8, 1229.0, 1226.85, 1422.2, 1471.3, 1324.35, 1181.6, 1352.55, 1274.85, 1225.8, 1257.45, 965.75, 1171.55, 1193.5, 1089.55, 1107.55, 1040.85, 1209.9, 1245.9, 1268.0, 1178.25, 1310.1, 1117.2, 1290.8, 1265.9, 1329.9, 1171.35]

alpha00005_4 = [1416.5, -57.2, -122.4, 149.3, 392.7, 293.95, 644.4, 1424.5, 1045.95, 1361.95, 1414.25, 490.95, 1166.25, 587.15, 975.45, 1262.1, 1191.65, 1118.75, 1235.3, 1403.0, 1402.85, 1370.5, 875.35, 1200.05, 951.05, 966.4, 750.6, 857.0, 932.8, 1231.75, 1188.6, 996.25, 1146.5, 878.7, 1008.85, 840.7, 956.05, 1063.55, 956.55, 1227.65, 1079.2, 1409.0, 1040.2, 1036.8, 1027.85, 942.85, 1146.0, 1314.3, 1423.7, 1071.85, 1369.0, 1341.6, 1233.95, 1416.0, 1105.45, 1288.45, 1254.3, 1362.55, 1060.85, 1168.05, 1272.9, 1284.4, 1331.25, 1420.25, 1416.95, 1405.1, 1467.2, 1271.2, 1243.55, 1353.35, 1359.25, 1242.1, 990.2, 1257.8, 1464.5, 1312.95, 1360.55, 1272.1, 1239.0, 1405.75, 1249.05, 473.45, 1297.2, 1339.4, 1162.45, 1321.0, 1307.35, 1317.1, 1194.1, 1001.75, 1293.35, 1075.9, 1105.4, 1180.25, 1024.9, 1414.3]

sim9999_1 = [-400.5, -389.15, -394.8, -292.5, -297.75, -262.6, -65.7, 377.5, 373.95, 1001.95, 1069.5, 750.9, 170.55, 633.2, 1092.2, 672.0, 616.6, 1013.7, 1221.2, 1294.8, 808.45, 1392.05, 1164.3, 1003.9, 928.75, 914.8, 883.5, 581.2, 660.35, 1006.2, 735.3, 1248.7, 1154.6, 701.1, 1062.55, 1103.8, 666.25, 933.45, 979.45, 586.85, 971.55, 831.2, 1252.4, 1055.65, 1241.1, 1227.85, 1116.65, 1254.85, 1300.25, 1258.05, 1478.65, 1341.55, -49.8, 1093.2, 1003.9, 1336.4, 1129.95, 1419.25, 850.55, 1473.8, 1204.35, 1105.8, 1296.9, 1079.5, 1332.65, 1064.85, 1304.05, 1165.45, 1199.3, 871.25, 1334.45, 1216.0, 1129.8, 1059.3, 1265.05, 1090.8, 737.7, 1250.25, 1424.7, 1105.15, 1293.05, 1207.4, 1397.5, 1195.6, 1287.05, 1188.4, 1041.55, 853.85, 1144.7]

sim9999_2 = [-358.0, -379.9, -410.15, -250.4, -253.4, -222.95, -279.75, -233.1, -200.3, -221.5, -296.85, -152.0, 253.7, 185.4, 1001.8, 1340.5, 1392.1, 1314.2, 1451.3, 1270.45, -41.8, 820.0, 1408.05, 818.15, 1251.55, 643.2, 1167.8, 1178.1, 1060.9, 1295.85, 1197.0, 1465.05, 1362.0, 1187.9, 1261.8, 1338.2, 988.5, 1015.05, 1476.9, 745.2, 891.85, 1194.35, 1276.25, 1471.65, 1340.2, 1293.15, 1476.55, 1218.75, 988.15, 1469.3, 1230.1, 597.95, 422.35, 1214.1, 1327.95, 1369.3, 1258.65, 1233.0, 917.4, 1073.0, 1021.7, 1416.2, 1476.95, 1476.2, 1419.0, 796.85, 1287.1, 1337.1, 1111.0, 1093.25, 908.6, 1131.95, 1318.8, 1335.9, 1361.65, 1091.35, 1473.15, 1326.6, 1000.6, 1276.45, 1183.8, 1004.5, 1102.0, 1311.2, 1414.25, 1469.4, 715.15, 1414.6, 815.45, 727.45, 1027.3, 1141.8, 1393.9, 41.7, 308.7, 875.5, 1346.5, 519.0, 1004.75, -262.05, -262.75, -357.8, -219.65, 121.0]

sim9999_3 = [-393.25, -388.05, -365.2, -238.5, -247.6, -249.85, 38.1, 595.5, 1094.25, 1006.35, 1153.15, 722.6, 938.7, 1204.55, 1148.5, 1314.35, 820.1, 1377.7, 767.75, 939.7, 1289.6, 1360.2, 1469.55, 1298.4, 525.15, 1204.85, 1283.65, 968.2, 1046.8, 1193.8, 405.45, 1332.6, 1325.5, 293.2, 599.9, 1465.3, 918.65, 1119.85, 1418.5, 1380.2, 857.8, 1236.5, 1466.35, 1468.3, 1340.2, 1246.85, 1002.65, 1251.3, 1415.25, 1474.9, -63.2, 51.85, 736.3, 1424.65, 1070.3, 1428.55, 1422.4, 1400.2, 1389.0, 1204.8, 957.7, 1123.15, 1408.1, 1142.8, 1182.25, 677.35, 948.85, 785.35, 1477.1, 1075.05, 1465.95, 1427.5, 1477.55, 1412.3, 1055.35, 1374.9]

sim9999_4 = [-400.5, -389.15, -394.8, -292.5, -297.75, -262.6, -65.7, 377.5, 373.95, 1001.95, 1069.5, 750.9, 170.55, 633.2, 1092.2, 672.0, 616.6, 1013.7, 1221.2, 1294.8, 808.45, 1392.05, 1164.3, 1003.9, 928.75, 914.8, 883.5, 581.2, 660.35, 1006.2, 735.3, 1248.7, 1154.6, 701.1, 1062.55, 1103.8, 666.25, 933.45, 979.45, 586.85, 971.55, 831.2, 1252.4, 1055.65, 1241.1, 1227.85, 1116.65, 1254.85, 1300.25, 1258.05, 1478.65, 1341.55, -49.8, 1093.2, 1003.9, 1336.4, 1129.95, 1419.25, 850.55, 1473.8, 1204.35, 1105.8, 1296.9, 1079.5, 1332.65, 1064.85, 1304.05, 1165.45, 1199.3, 871.25, 1334.45, 1216.0, 1129.8, 1059.3, 1265.05, 1090.8, 737.7, 1250.25, 1424.7, 1105.15, 1293.05, 1207.4, 1397.5, 1195.6, 1287.05, 1188.4, 1041.55, 853.85, 1144.7, 811.9, 1357.1]

sim9995_1 = [-384.5, -381.05, -364.15, -356.5, -366.4, -365.65, -362.8, -361.3, -353.0, -365.35, -360.25, -352.0, -364.45, -366.35, -356.15, -368.75, -367.5, -377.6, -363.1, -360.75, -377.9, -377.15, -340.1, -280.15, -181.4, -192.1, -270.75, -479.0, -308.45, -319.4, -92.55, 83.2, 9.3, 33.65, 95.35, -91.35, -177.95, -108.05, -199.45, -74.8, -96.0, 11.55, -87.75, -49.9, -164.5, -26.1, -108.65, -165.3, -157.65, -195.55, -39.05, -137.25, -128.4, -117.15, -239.2, -136.85, -276.25, -155.25, -272.5, -195.05, -83.6, -88.1, -247.4, -173.1, -199.5, -144.65, -132.15, -134.15, -170.1, -181.3, -236.85, -180.7, -269.75, -30.2, -244.95, -301.55, -155.6, -206.6, -139.6, -150.45, -117.8, -194.3, -174.4, -128.25, -83.05, -197.75, -114.85, -301.9, -120.05, -304.55, -220.2, -213.2, -128.4, -15.05, -206.25, -193.3, -265.15, -100.1, -189.6, -279.05, -387.05, -326.3, -237.2, -277.3]

sim9995_2 = [-390.5, -394.9, -328.5, -244.05, 205.3, 569.85, 579.0, 728.6, 853.9, 591.55, 774.0, 1307.8, 992.8, 1065.55, 989.5, 977.95, 1007.85, 1010.7, 1001.65, 878.4, 984.65, 763.55, 765.1, 991.6, 947.1, 1029.0, 973.0, 1312.9, 1060.35, 1113.3, 1155.05, 787.3, -99.05, 296.7, 1227.15, 1250.0, 452.1, 1423.0, 756.1, 1159.8, 1090.9, 497.3, 760.65, 969.0, 1197.45, 872.2, 1388.5, 1140.8, 970.6, 1286.2, 845.3, 1160.05, 1269.65, 1477.5, 1402.3, 1131.85, 1297.45, 678.1, 1367.05, 1142.75, 516.45, 156.45, 565.7, 876.55, 1193.0, 923.55, 1094.35, 1088.45, 1164.7, 1030.7, 1012.5, 1216.35, 711.7, 1274.15, 1171.15, 1321.9, 1345.4, 1388.85, 1367.15, 511.45, 382.1, 209.7, 1173.35, 1177.15, 1058.9, 952.95, 1384.5, 1371.55, 1260.25, 1321.65, 482.45, 1398.5, 1045.0, 993.7, 1427.7, 1112.1, 430.65, 1248.6, 1462.65, 1476.25, 1477.9, 1458.8, 1358.0, 1412.05]

sim9995_3 = [-461.7, -460.25, -386.35, -251.9, -119.25, 512.75, 1056.1, 1101.35, 1048.85, 1038.3, 1152.3, 967.35, 1126.2, 1056.55, 1080.0, 1196.15, 1347.0, 877.25, 1071.0, 1253.1, 1338.15, 1289.55, 1458.1, 1288.75, 823.4, 1292.35, 1274.5, 1368.9, 1160.8, 1189.2, 1026.05, 1342.45, 1061.1, 1094.3, 1246.55, 733.4, 1240.3, 663.75, 1231.5, 592.5, 975.2, 619.35, 886.55, 1032.85, 1414.75, 1094.5, 1363.4, 408.05, 952.45, 1081.5, 1464.05, 1366.45, 1248.75, 1062.85, 1264.85, 1088.6, 1017.2, 1464.15, 1328.95, 1307.75, 1351.95, 52.4, -155.7, 1002.3, 1095.5, 975.45, 1469.7, 1475.5, 1309.75, 1475.5, 1422.75, 1391.9, 1018.55, 1123.95, 1420.7, 1094.75, 903.0, 1265.6, 1354.85, 437.0, 309.55, 997.75, 1344.45, 1271.7, 1314.05, 1314.7, 1069.05, 1224.2, 1265.35, 1077.4, 1029.65, 1161.9, 859.8, 1148.5, 1311.55, 1003.2, 897.2, 1266.2, 854.25]

sim9995_4 = [-344.3, -169.6, -179.3, 462.4, 391.5, 1172.35, 1097.8, 1335.6, 1279.0, 1045.9, 1191.8, 1217.05, 1125.35, 1154.0, 1114.15, 1282.1, 1084.85, 963.65, 860.25, 717.9, 871.55, 800.05, 1239.0, 1364.95, 1191.4, 1093.45, 533.45, 1016.9, 1458.7, 1169.45, 1307.5, 1473.6, 988.45, 1290.75, 970.75, 1158.05, 604.3, 1267.3, 1199.85, 1471.9, 1474.9, 499.05, 505.9, 415.05, 1323.75, 1250.65, 921.05, 1320.55, -140.9, 62.25, -2.55, -319.1, -212.75, -171.9, -28.85, -161.35, -251.3, 39.7, -161.2, -199.3, 234.1, 1415.7, 1269.6, 1421.35, 1473.9, 1316.0, 1470.3, 1469.05, 1425.15, 1420.15, 567.9, 767.95, 1394.65, 1473.8, 1043.5, 1348.7, 1285.7, 1476.4, 1131.6, 1267.7, 1308.3, 548.75, 1199.8, 880.85, 1091.65, 1049.35, 1069.4, 1306.4, 1286.1, 1392.55, 823.2, 1277.3, 1471.7, 1266.8, 1132.15, 1051.2, 791.2, 1336.35, 1093.8, 1380.15, 1484.6, 1347.25, 1478.0, 1424.85]

sim999_1 = [-394.4, -385.55, -342.4, -350.85, -336.05, -345.55, -337.3, -332.0, -336.0, -323.3, -365.85, -326.9, -299.9, -265.85, -306.2, -318.8, -312.9, -299.9, -292.75, -285.8, -287.65, -319.2, -298.8, -325.8, -327.85, -333.0, -316.15, -293.4, -330.9, -274.0, -283.75, -335.3, -297.4, -295.2, -260.9, -296.6, -290.85, -304.75, -276.05, -275.45, -303.9, -258.95, -243.95, -319.7, -282.65, -297.6, -300.55, -322.8, -251.5, -274.35, -289.95, -264.65, -263.2, -249.35, -243.45, -216.15, -272.9, -299.05, -224.85, -281.8, -279.25, -191.7, -292.7, -276.85, -224.3, -209.0, -167.35, -18.9, -158.85, 26.35, 348.2, 267.65, 101.75, 120.6, 123.4, 92.2, 735.35, 811.35, 1117.25, 1014.55, 1197.0, 1169.1, 1037.35, 1165.0, 401.7, 722.95, 1072.35, 1000.75, 742.25, 781.15, 1116.3, 1382.45, 1032.8, 1407.05, 1297.6, 772.7, 772.35, 1065.2, 1394.15, 1397.75, 1473.05, 1467.95, 1397.9, 1471.2]

sim999_2 = [-395.95, -393.45, -389.15, -438.55, -423.1, -414.7, -432.15, -420.4, -409.35, -419.35, -279.95, -295.15, -150.1, -154.4, -316.9, -148.05, -118.7, -288.45, -152.1, -219.1, -114.45, -180.3, -312.0, -249.15, -264.5, -214.95, -281.25, -191.5, -317.3, -227.5, -329.6, -341.25, -353.05, -272.35, -266.5, -261.55, -306.1, -220.3, -295.35, -271.25, -233.6, -294.35, -225.25, -71.75, -166.7, -49.7, 38.65, 375.85, 817.35, 624.5, 332.0, 648.25, 450.4, 853.75, 736.8, 578.85, 389.0, 256.05, 522.85, 422.75, 78.95, 718.6, 684.55, 751.7, 1023.9, 1198.1, 566.4, 321.5, 487.9, 399.25, 222.2, 1136.5, 696.2, 746.35, 498.75, 361.05, 505.25, 473.35, 114.35, 477.2, 558.9, 549.75, -133.7, -157.1, 211.9, 395.55, 653.3, 618.75, 604.15, 758.9, 432.45, 262.9, 730.05, 743.1, 660.15, 409.8, 886.6, 696.8, 1040.45, 82.15, 219.85, 329.35, 410.85, 207.4]

sim999_3 = [-401.65, -345.7, -293.05, -255.25, -190.45, -324.0, -334.4, -334.6, -259.4, -102.95, -122.15, -107.65, -275.65, -132.5, -308.5, -225.35, -158.35, -278.45, -240.9, -271.75, -211.85, -255.6, -275.4, -194.9, -210.25, -219.4, -183.75, -168.2, -238.1, -148.95, -178.25, -183.15, -312.3, -335.3, -258.75, -256.65, -161.8, -241.1, -252.5, -243.65, -245.05, -250.55, -278.15, -214.0, -198.25, -185.2, -141.25, -264.45, -200.5, -225.4, -161.8, -187.35, -201.05, -121.1, -209.5, -204.25, -204.65, -261.3, -142.8, -158.5, -172.4, -140.05, -209.2, -225.6, -185.55, -207.25, -157.45, -176.4, -194.9, -215.55, -113.4, -257.1, -148.5, -150.5, -156.05, -137.5, -232.85, -221.45, -221.1, -219.75, -205.9, -112.5, -170.1, -191.55, -220.8, -186.6, -120.75, -99.65, -253.1, -260.0, -210.35, -109.55, -226.8, -152.75, -186.55, -146.5, -206.6, -127.7, -170.0, -129.5, -170.25, -155.55, -169.05, -273.2]


# plt.plot(runningAvg(averageVector([alpha0001_1, alpha0001_2, alpha0001_3, alpha0001_4]), 1), color='k')
plt.plot(runningAvg(averageVector([sim999_1, sim999_2, sim999_3])), color='b')
plt.plot(runningAvg(averageVector([sim9999_1, sim9999_2, sim9999_3, sim9999_4])), color='y')
plt.plot(runningAvg(averageVector([sim9995_2, sim9995_3, sim9995_4])), color='k')
# plt.plot(runningAvg(averageVector([alpha0002_1, alpha0002_2, alpha0002_3, alpha0002_4]), 15), color='b')
# plt.plot(runningAvg(averageVector([alpha0005_1, alpha0005_2, alpha0005_3, alpha0005_4])), color='y')
# plt.plot(runningAvg(averageVector([alpha00002_1, alpha00002_2, alpha00002_4])), color='c')
# plt.plot(runningAvg(averageVector([alpha00005_1, alpha00005_2, alpha00005_3, alpha00005_4])), color='r')
# plt.plot(runningAvg(averageVector([eps9995_1, eps9995_2, eps9995_3])), color='k')
# plt.plot(runningAvg(averageVector([eps9999_1, eps9999_2, eps9999_3])), color='y')
plt.show()


'''
grep asked trainQ.01_1 |  awk -F ' ' '{print $2}' | awk NF=NF RS= OFS=", "
'''
