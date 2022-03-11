﻿using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkLearning
{
    // Spiral Dataset
    class Dataset
    {
        public static double[,] input = { { 0, 0 }, { 0.00822486, 0.00586363 }, { 0.00672537, 0.0190497 }, { 0.01093163, 0.02826257 }, { 0.00862362, 0.03947302 }, { 0.03232221, 0.03880767 }, { -0.00058822, 0.06060321 }, { 0.04399159, 0.05535549 }, { 0.04602689, 0.06641891 }, { 0.08159491, 0.04008409 }, { 0.07074513, 0.07209831 }, { 0.10868695, 0.02308301 }, { 0.11034471, 0.05016397 }, { 0.05785842, 0.11787935 }, { 0.09167459, 0.10767418 }, { 0.11961344, 0.09300251 }, { 0.12766982, -0.09909692 }, { 0.14371956, -0.09397593 }, { 0.1817185, -0.00601974 }, { -0.07143852, -0.1781278 }, { 0.12267344, 0.16050978 }, { 0.16152922, -0.1374908 }, { 0.12484985, -0.18383479 }, { 0.21342489, -0.0917818 }, { 0.23184287, -0.07084065 }, { 0.14451406, -0.20708619 }, { -0.12878917, -0.22887967 }, { 0.03078162, -0.27098461 }, { -0.21892035, -0.17906904 }, { 0.16877092, -0.2394242 }, { 0.03711258, -0.3007491 }, { -0.14291109, -0.27861737 }, { 0.06194992, -0.3172402 }, { 0.02770095, -0.33218032 }, { 0.223087, -0.26111174 }, { -0.07343605, -0.34582422 }, { -0.30588758, -0.19663213 }, { -0.17308547, -0.33124167 }, { -0.14488769, -0.35544263 }, { -0.04741384, -0.39107566 }, { -0.2883446, -0.2830301 }, { -0.32859931, -0.2520627 }, { -0.42360731, -0.02320514 }, { -0.38282812, -0.20517517 }, { -0.2544983, -0.36436449 }, { -0.44861793, -0.07316778 }, { -0.46463997, 0.00245596 }, { -0.39411736, -0.26468221 }, { -0.48429112, -0.02324151 }, { -0.49329114, -0.04048278 }, { -0.33711518, 0.37607096 }, { -0.45089057, 0.24915613 }, { -0.37857744, 0.36410073 }, { -0.44824881, 0.29270533 }, { -0.49334955, 0.23265184 }, { -0.46957472, 0.29688644 }, { -0.14655527, 0.54634138 }, { -0.36739844, 0.44330032 }, { -0.44716588, 0.37851415 }, { -0.13752408, 0.57987496 }, { -0.30477534, 0.52385251 }, { -0.21628869, 0.57695263 }, { -0.07884614, 0.62127946 }, { 0.50960408, 0.38113299 }, { 0.34277876, 0.54810516 }, { 0.27502403, 0.5961881 }, { -0.39396555, 0.53780627 }, { 0.37881716, 0.56081374 }, { 0.36809474, 0.57990935 }, { 0.05651581, 0.69467454 }, { 0.20190741, 0.67762997 }, { 0.65921702, 0.28243262 }, { 0.72665242, 0.0300314 }, { 0.72725757, 0.12172289 }, { 0.42875122, 0.61228334 }, { 0.7348449, -0.18418469 }, { 0.76731173, -0.02367138 }, { 0.57292406, 0.52601929 }, { 0.20454446, -0.76086434 }, { 0.69719842, -0.38818311 }, { 0.80206892, 0.09838717 }, { 0.47354723, -0.66721399 }, { 0.81490538, -0.14826218 }, { -0.17189176, -0.82057339 }, { 0.56623232, -0.63190783 }, { 0.48680626, -0.70724065 }, { 0.59053863, -0.63708791 }, { 0.48358167, -0.73376897 }, { 0.66019687, -0.59520043 }, { 0.1869207, -0.87934265 }, { 0.70331498, -0.5760159 }, { 0.25948472, -0.8818058 }, { 0.12429904, -0.9209425 }, { 0.03347322, -0.93879738 }, { 0.51558778, -0.79731418 }, { -0.05312675, -0.95812419 }, { -0.64582327, -0.7233426 }, { 0.32390827, -0.92470942 }, { -0.51023185, -0.84827087 }, { -0.5912785, -0.80646744 }, { 0, 0 }, { -0.00925558, -0.00404531 }, { -0.01018928, -0.0174442 }, { -0.02435745, -0.01802743 }, { -0.03535052, -0.01956598 }, { -0.04902964, -0.01211836 }, { -0.05655931, -0.02177475 }, { -0.06978185, 0.01140101 }, { -0.08058063, -0.00605866 }, { -0.08076545, 0.04173014 }, { -0.0647732, 0.07750789 }, { -0.10439236, 0.03805147 }, { -0.08824578, 0.0830967 }, { -0.08422077, 0.10074721 }, { -0.13468621, -0.04309969 }, { -0.14322159, -0.04944106 }, { -0.15524191, 0.04494145 }, { 0.01464171, 0.17109181 }, { -0.02678946, 0.17983375 }, { -0.1620948, 0.10275335 }, { -0.12678959, 0.15727862 }, { -0.08541825, 0.19416264 }, { -0.08497299, 0.20533462 }, { 0.00427293, 0.23228393 }, { 0.02834523, 0.24076142 }, { 0.02652984, 0.2511278 }, { 0.14573216, 0.2184827 }, { 0.17886329, 0.20588368 }, { 0.03092888, 0.28113207 }, { 0.05013066, 0.28860784 }, { 0.21074946, 0.21774304 }, { 0.09642179, 0.29791619 }, { 0.29279898, 0.13692293 }, { 0.2848824, 0.17306972 }, { -0.0794084, 0.3341279 }, { 0.30448165, 0.17966126 }, { 0.02489723, 0.36278304 }, { 0.30389239, 0.21755239 }, { 0.38198834, 0.03764057 }, { 0.38366604, -0.08937904 }, { 0.32810824, -0.23578301 }, { 0.34680448, 0.22636201 }, { 0.38539873, -0.17733993 }, { 0.43403536, -0.01635617 }, { 0.44328989, -0.0320146 }, { 0.44832688, -0.07493051 }, { 0.41903562, -0.20076227 }, { -0.14052617, -0.45347278 }, { 0.35613582, -0.32900658 }, { 0.48730817, 0.08663575 }, { 0.15725254, -0.47994547 }, { 0.3231358, -0.40120362 }, { 0.19593312, -0.48734016 }, { 0.41751968, -0.33508316 }, { 0.2816187, -0.46713121 }, { 0.22579805, -0.50759946 }, { 0.38980494, -0.40990177 }, { -0.34643092, -0.45987216 }, { 0.08774399, -0.57925062 }, { 0.3534925, -0.47980297 }, { -0.14223084, -0.58913483 }, { -0.1948301, -0.584548 }, { 0.21195755, -0.58930372 }, { -0.04859032, -0.63450584 }, { -0.61177488, -0.20892113 }, { -0.58555657, -0.29698816 }, { -0.47729328, -0.46544126 }, { -0.55263387, -0.39065368 }, { -0.63789582, -0.25471065 }, { -0.4646369, 0.51949909 }, { -0.65040271, 0.2773541 }, { -0.71287519, -0.07838513 }, { -0.71544811, 0.13061248 }, { -0.70241419, -0.22435313 }, { -0.66788436, 0.33563221 }, { -0.68563956, 0.32221022 }, { -0.40737975, 0.65066839 }, { -0.72173261, 0.28989706 }, { -0.7044547, 0.3528407 }, { -0.78829905, -0.12392079 }, { -0.70769919, 0.39007237 }, { -0.08872228, 0.81335715 }, { -0.82814519, -0.01509914 }, { -0.63660845, 0.54554298 }, { 0.13093194, 0.83832176 }, { -0.20430045, 0.83392506 }, { -0.65255354, 0.57340279 }, { -0.50282329, 0.72071969 }, { -0.31865164, 0.82980997 }, { -0.34031528, 0.83208674 }, { 0.2599486, 0.87113317 }, { -0.2840963, 0.87418709 }, { 0.08517148, 0.92538163 }, { 0.67045656, 0.65798858 }, { 0.94949384, -0.00145247 }, { 0.26873555, 0.92119792 }, { 0.96944052, -0.02230006 }, { 0.93352465, 0.29754966 }, { 0.58868685, 0.79583152 }, { 0.06514654, 0.99787571 }, { 0, 0 }, { 0.00776586, 0.00645924 }, { 0.01632069, 0.01190616 }, { 0.0111219, 0.02818824 }, { 0.03531827, 0.01962413 }, { 0.03021805, 0.04046763 }, { 0.05734761, -0.01960476 }, { 0.01045362, -0.06993005 }, { 0.0804955, -0.00710068 }, { 0.05999501, -0.06830126 }, { 0.08914932, -0.04749147 }, { 0.09338185, -0.06021219 }, { 0.07882211, -0.09208395 }, { 0.02737343, -0.12842832 }, { 0.06260695, -0.12680035 }, { 0.12364048, -0.08757781 }, { -0.02168584, -0.16015464 }, { 0.05891575, -0.1612939 }, { 0.04914761, -0.1750496 }, { 0.01239212, -0.1915187 }, { -0.12456327, -0.15904765 }, { -0.05986625, -0.20349801 }, { -0.17532265, -0.13654554 }, { -0.11196768, -0.2035616 }, { -0.08767209, -0.22601575 }, { -0.11177338, -0.22644142 }, { -0.03174535, -0.26070057 }, { 0.03403077, -0.27059577 }, { -0.19587504, -0.20402158 }, { -0.18214772, -0.22941181 }, { 0.15291358, -0.26161957 }, { -0.30788274, 0.05709146 }, { -0.06053805, -0.31751264 }, { -0.31477759, 0.10966393 }, { -0.29243178, -0.18008553 }, { -0.27320599, 0.22437854 }, { -0.18643054, 0.31220996 }, { -0.36734359, -0.0688354 }, { -0.35731172, 0.14021497 }, { -0.36070855, -0.15835906 }, { -0.22992432, 0.33224006 }, { -0.32077621, 0.26194605 }, { -0.31505331, 0.28411801 }, { -0.28255466, 0.32987434 }, { -0.2157766, 0.38855028 }, { -0.39345337, 0.22760935 }, { -0.28019642, 0.37065658 }, { -0.1442144, 0.45231336 }, { 0.27201397, 0.40135577 }, { 0.20823209, 0.44901492 }, { 0.24139015, 0.44362914 }, { -0.09467005, 0.50637799 }, { -0.09639654, 0.51633121 }, { 0.07189928, 0.53050344 }, { 0.11220925, 0.53378811 }, { -0.15689146, 0.53294188 }, { 0.33959451, 0.45237476 }, { 0.50583058, 0.2750131 }, { 0.50771027, 0.29233639 }, { -0.22883147, 0.55027629 }, { 0.43180815, 0.42526601 }, { 0.58824034, 0.18338058 }, { 0.55375349, 0.29250975 }, { 0.52618312, 0.35789663 }, { 0.60535101, -0.22686272 }, { 0.5881126, 0.29189387 }, { 0.65318711, -0.13338309 }, { 0.61462287, -0.28328999 }, { 0.61340579, -0.30906622 }, { 0.35579001, -0.59931646 }, { 0.66067994, -0.25189482 }, { 0.56874604, -0.43687895 }, { 0.6442715, -0.33740162 }, { 0.66683021, -0.31473401 }, { 0.32559713, -0.67283357 }, { 0.64479096, -0.39770044 }, { 0.37362549, -0.67062032 }, { 0.05173887, -0.776055 }, { -0.02764684, -0.78739357 }, { 0.35917066, -0.71257855 }, { 0.60803862, -0.53224395 }, { -0.01939911, -0.81795181 }, { -0.50832761, -0.65395373 }, { -0.50115733, -0.67210772 }, { -0.53687845, -0.65702973 }, { -0.17969901, -0.83957009 }, { -0.16331474, -0.85319703 }, { -0.32815556, -0.81521903 }, { -0.30383696, -0.83534817 }, { -0.39958572, -0.80530373 }, { 0.2053967, -0.88558369 }, { -0.26927792, -0.87886471 }, { -0.33629028, -0.86631068 }, { -0.53514199, -0.77206478 }, { -0.5550179, -0.77038678 }, { -0.95886743, -0.03738535 }, { -0.96548905, 0.09023922 }, { -0.97955668, -0.02174366 }, { -0.95524905, -0.25961366 }, { -0.94765869, 0.31928515 } };
        public static double[] trueS = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };
        public static double[,] testin = { { -0, 0 }, { 0.00182466, 0.00993484 }, { 0.01174838, 0.01643463 }, { 0.00880484, 0.02899566 }, { 0.03530393, 0.01964991 }, { 0.04187354, 0.02823769 }, { 0.06043173, 0.00459352 }, { 0.06146992, 0.03494194 }, { 0.0694911, 0.04124237 }, { 0.08888982, 0.0190542, }, { 0.07320419, 0.06960019 }, { 0.04143688, 0.10309541 }, { 0.11585441, 0.03563894 }, { 0.04248399, 0.12425075 }, { 0.13251494, 0.04937358 }, { 0.14987185, 0.02225463 }, { 0.16161509, -0.00058813 }, { 0.14836332, 0.08645873 }, { 0.13404263, -0.12284309 }, { 0.16429147, -0.09920328 }, { 0.12493393, -0.15875665 }, { 0.13872899, -0.16046706 }, { 0.17334043, -0.13905326 }, { -0.03752177, -0.2292732, }, { 0.24241274, 0.00236148 }, { 0.23266411, -0.09816523 }, { 0.0570453, -0.25635598 }, { -0.01923938, -0.27204781 }, { 0.05218056, -0.27797307 }, { -0.16653531, -0.24098456 }, { 0.06225685, -0.2965661, }, { -0.1125362, -0.29221024 }, { -0.06089664, -0.31744407 }, { 0.01428908, -0.33302693 }, { 0.03572966, -0.3415707, }, { -0.0131372, -0.35329118 }, { -0.20153251, -0.30268144 }, { -0.37352436, 0.0126165, }, { -0.27931957, -0.26327264 }, { -0.244115, -0.30918621 }, { -0.40009051, -0.05635808 }, { -0.34888467, -0.22314256 }, { -0.3990751, 0.14394686 }, { -0.42355932, -0.09618588 }, { -0.44364331, -0.02667362 }, { -0.23870438, -0.38682268 }, { -0.46179719, -0.05137793 }, { -0.41253736, 0.23494275 }, { -0.44714089, -0.18746488 }, { -0.42410992, 0.25515834 }, { -0.442673, 0.24313912 }, { -0.51397305, -0.03482512 }, { -0.0292412, 0.52443795 }, { -0.52455764, 0.10697051 }, { -0.39847388, 0.37247715 }, { -0.20604287, 0.51593441 }, { -0.55811206, -0.09207756 }, { -0.49332195, 0.29686738 }, { 0.16973308, 0.56073253 }, { -0.36210205, 0.47333914 }, { -0.28664691, 0.53398783 }, { 0.12438218, 0.60347677 }, { 0.5982363, 0.18525172 }, { -0.1446052, 0.61971608 }, { 0.15118723, 0.62853716 }, { -0.25825509, 0.60364126 }, { -0.00547204, 0.66664421 }, { 0.63703686, 0.22846997 }, { 0.36007936, 0.58492003 }, { 0.23228747, 0.65712197 }, { 0.39299046, 0.58779884 }, { 0.47633905, 0.53613093 }, { 0.63854082, 0.34812532 }, { 0.71889312, 0.16405095 }, { 0.74710605, 0.02347436 }, { 0.71152038, 0.26011493 }, { 0.75669737, -0.12937042 }, { 0.63843459, 0.44422916 }, { 0.75652509, 0.22005174 }, { 0.73598349, -0.30838298 }, { 0.80663671, -0.04828887 }, { -0.09512618, -0.81263306 }, { 0.66991112, -0.48710527 }, { 0.5766284, -0.60859441 }, { 0.79452653, -0.29774844 }, { 0.59268975, -0.62119927 }, { 0.39753207, -0.77238924 }, { 0.70639817, -0.5227521, }, { 0.31668121, -0.83056395 }, { 0.51906448, -0.73399925 }, { -0.47722711, -0.77375744 }, { 0.33238501, -0.85699124 }, { 0.03771941, -0.92852711 }, { -0.67849939, -0.64969188 }, { -0.92204711, -0.22664904 }, { -0.02547799, -0.95925767 }, { -0.61517582, -0.7495805, }, { -0.88155857, -0.42761965 }, { 0.28119342, -0.94912079 }, { -0.04327805, -0.99906307 }, { -0, -0 }, { -0.00826067, -0.00581306 }, { -0.01498883, -0.01354461 }, { -0.02084988, -0.02198991 }, { -0.02942022, -0.02769363 }, { -0.04682064, -0.01893641 }, { -0.02380598, -0.05573482 }, { -0.06619875, -0.02484382 }, { -0.03750023, -0.07157987 }, { -0.09062207, -0.0072183, }, { -0.09579485, -0.03203728 }, { -0.11104437, -0.00385048 }, { -0.11316889, 0.04341867 }, { -0.11660368, 0.06038808 }, { -0.1345889, 0.04340261 }, { -0.03100873, 0.14830812 }, { -0.11495343, 0.11360234 }, { -0.14114989, 0.09779313 }, { 0.0058625, 0.18172364 }, { -0.08485345, 0.172142, }, { -0.05815355, 0.19346919 }, { -0.11696166, 0.17696152 }, { -0.03108166, 0.22003783 }, { -0.08658023, 0.21558745 }, { -0.09404207, 0.22344038 }, { -0.04156862, 0.24908041 }, { -0.11816202, 0.23454273 }, { -0.04329003, 0.26926964 }, { 0.16412371, 0.23033724 }, { 0.12223746, 0.26620589 }, { 0.09423443, 0.28800562 }, { 0.09831624, 0.29729638 }, { 0.31763173, 0.0599101, }, { 0.2869911, 0.16955004 }, { 0.07855464, 0.33432965 }, { 0.33645617, 0.1085564, }, { 0.21265469, 0.29497354 }, { 0.37016237, 0.05156979 }, { 0.2481334, 0.29285102 }, { 0.21838759, -0.32786446 }, { 0.3978773, -0.07030148 }, { 0.40540563, -0.08461313 }, { 0.42349601, -0.02515481 }, { 0.39176409, -0.1875503, }, { 0.41986545, -0.14575278 }, { 0.45109877, -0.05587014 }, { 0.46156675, 0.05340853 }, { 0.35717779, -0.3127446, }, { 0.47765321, 0.08321938 }, { 0.33230772, -0.36680592 }, { 0.14191469, -0.48470221 }, { 0.19090513, -0.4784729, }, { 0.25889226, -0.45701752 }, { 0.3182514, -0.43048746 }, { 0.07780174, -0.53987735 }, { 0.20499735, -0.51635071 }, { 0.12513976, -0.55164064 }, { 0.41694868, -0.39705237 }, { -0.47709615, -0.34001404 }, { 0.1971558, -0.56240326 }, { 0.07803183, -0.60101622 }, { -0.21718965, -0.57661408 }, { -0.1220838, -0.61424785 }, { 0.22481125, -0.59533065 }, { -0.58747243, -0.26980119 }, { -0.58344834, -0.30110879 }, { -0.01388108, -0.66652214 }, { -0.4840264, -0.47300416 }, { -0.68484588, 0.05267559 }, { -0.67280192, 0.18194599 }, { -0.62953732, -0.32191885 }, { -0.64209548, -0.31945057 }, { -0.72721863, 0.00887061 }, { -0.48913666, -0.5517838, }, { -0.7104224, -0.23241883 }, { -0.73221198, -0.19438789 }, { -0.51340925, 0.57073511 }, { -0.74722724, 0.21584652 }, { -0.02732583, 0.78740478 }, { -0.62143077, 0.5005952, }, { -0.10111055, 0.80173016 }, { -0.55389945, 0.60217679 }, { -0.7202793, 0.40896231 }, { -0.13497264, 0.82744779 }, { -0.66036317, 0.53277295 }, { -0.04142086, 0.85758614 }, { -0.13294896, 0.85845294 }, { -0.5425966, 0.69127206 }, { -0.56348167, 0.68746772 }, { 0.04935991, 0.8976338, }, { 0.51153804, 0.75151521 }, { -0.41544418, 0.81995117 }, { -0.0596695, 0.92737527 }, { -0.08066033, 0.93592462 }, { 0.949089, -0.02776187 }, { 0.93135791, 0.23107757 }, { 0.84960367, 0.46742466 }, { 0.72700482, 0.65686229 }, { 0.98251115, 0.12071394 }, { 0.30479483, 0.95241804 }, { 0, 0 }, { 0.0073821, 0.00689457 }, { 0.01859973, 0.00788489 }, { 0.02922683, 0.00800413 }, { 0.01224678, -0.03850328 }, { 0.0496066, -0.00948394 }, { 0.06057763, -0.00185615 }, { 0.05976455, 0.03778476 }, { 0.07870053, 0.01833499 }, { 0.07306492, -0.05409233 }, { 0.05361496, -0.08560652 }, { 0.10752556, -0.0279988, }, { 0.12028209, -0.01498653 }, { 0.07437672, -0.10821849 }, { 0.04356357, -0.13453689 }, { 0.1270081, -0.0826183 }, { 0.00828668, -0.16140358 }, { 0.08016412, -0.15185684 }, { 0.11845489, -0.13793582 }, { -0.15848233, -0.10824198 }, { -0.04196913, -0.19761264 }, { -0.01919157, -0.21125125 }, { -0.12175534, -0.18589877 }, { -0.05545732, -0.22560711 }, { -0.15184578, -0.18897717 }, { -0.03957995, -0.24940415 }, { -0.14077842, -0.22170699 }, { -0.07320467, -0.26271894 }, { -0.22096638, -0.17653809 }, { -0.28827051, -0.05203543 }, { -0.2377499, -0.18788919 }, { -0.23193327, -0.21037628 }, { -0.30048506, -0.11911281 }, { -0.28029569, -0.18040354 }, { -0.32118749, -0.12159664 }, { -0.34406276, -0.08128998 }, { -0.31950274, -0.17363583 }, { -0.35524998, -0.1160908, }, { -0.34931533, 0.15909339 }, { -0.00441436, 0.39391466 }, { -0.26387432, 0.30597221 }, { -0.26707704, 0.31651693 }, { -0.16676501, 0.3900911, }, { -0.26582126, 0.3435015, }, { 0.05696659, 0.44077848 }, { 0.0949102, 0.44452629 }, { -0.28041481, 0.37049139 }, { -0.26120231, 0.39643224 }, { 0.40137428, 0.27198665 }, { 0.1980609, 0.4535933 }, { -0.08463723, 0.49790818 }, { 0.0329327, 0.51409777 }, { -0.06246742, 0.52152472 }, { 0.00635852, 0.53531577 }, { 0.011418, 0.54533503 }, { -0.03728166, 0.55430321 }, { 0.19007245, 0.53276619 }, { 0.38040452, 0.43219115 }, { 0.51311467, 0.28274303 }, { 0.31374604, 0.50668655 }, { 0.48805088, 0.35932687 }, { 0.18419078, 0.58798715 }, { 0.31988038, -0.53840637 }, { 0.42698701, -0.47184825 }, { 0.62747504, -0.15553655 }, { 0.61007714, 0.24266096 }, { 0.64532956, 0.16731469 }, { 0.67665692, -0.01224326 }, { 0.6336888, -0.26500396 }, { 0.13277863, -0.68420508 }, { 0.69010396, -0.15396593 }, { 0.68418525, -0.21500188 }, { 0.04692337, -0.72575741 }, { 0.60312266, -0.42422056 }, { 0.1858108, -0.72401163 }, { 0.37261193, -0.65960699 }, { 0.46045028, -0.61425822 }, { 0.66311858, -0.40646281 }, { 0.36401995, -0.69874349 }, { 0.03983442, -0.79698493 }, { 0.49454133, -0.63908017 }, { -0.01702255, -0.81800472 }, { 0.14255676, -0.8159228, }, { -0.49129822, -0.67934786 }, { 0.21528174, -0.82071938 }, { -0.15196974, -0.84502951 }, { -0.53533584, -0.68412895 }, { -0.81611025, -0.3259328, }, { -0.88887142, 0.00557318 }, { -0.64545113, -0.62576008 }, { -0.46857694, -0.77902627 }, { -0.71924256, -0.57236695 }, { -0.68705063, -0.62573699 }, { -0.74618761, -0.57067068 }, { -0.79917414, 0.51270006 }, { -0.95844025, -0.04708183 }, { -0.39488781, -0.88564995 }, { -0.97537291, -0.09301489 }, { -0.94974724, -0.27907023 }, { -0.9784195, 0.20662836 } };
        public static double[] testtrueS = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, };

    }
}

