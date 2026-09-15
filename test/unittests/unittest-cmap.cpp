// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author: Julian Melendez Delgado
//
// ENDLICENSE

#include "CharmmContext.h"
#include "CharmmCrd.h"
#include "CharmmPSF.h"
#include "CharmmParameters.h"
#include "CudaBondedForce.h"
#include "CudaEnergyVirial.h"
#include "Force.h"
#include "ForceManager.h"
#include "apo_test_helpers.h"
#include "catch.hpp"
#include "cuda_utils.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

// 1ubq has 5/6 cmaps of c36m its only missing Gly before Pro
// it has Ala, Gly, Pro, Ala before Pro and Pro-Pro
TEST_CASE("CMAP energy and gradients match OpenMM for 1UBQ") {
  auto parameters = std::make_shared<CharmmParameters>(
      apo_test::GetTopparDir() / "par_all36m_prot.prm");

  auto psf = std::make_shared<CharmmPSF>(
      apo_test::GetDataDir() / "1ubq.psf");

  auto coordinates = std::make_shared<CharmmCrd>(
      apo_test::GetDataDir() / "1ubq.crd");

  REQUIRE(psf->getNumCrossTerms() == 74);

  auto context = std::make_shared<CharmmContext>(psf, parameters);
  context->setBoxDimensions({50.0, 50.0, 50.0});
  context->setCoordinates(coordinates);

  context->calculateForces(false, true, false);

  const auto forceManager = context->getForceManager();
  REQUIRE(forceManager != nullptr);

  const std::map<std::string, double> energyComponents =
      forceManager->getEnergyComponents();

  REQUIRE(energyComponents.count("cmap") == 1);

  // OpenMM Reference platform using 1ubq.psf, 1ubq.crd, and
  // par_all36m_prot.prm without minimization.
  constexpr double expectedCmapEnergy = -40.314402494622;

  CHECK(energyComponents.at("cmap") ==
        Approx(expectedCmapEnergy).margin(1.0e-5));

  SECTION("Bonded forces sum to zero") {
    const auto bondedForces = forceManager->getBondedForcevalues();
    REQUIRE(bondedForces != nullptr);

    const std::size_t numAtoms = static_cast<std::size_t>(psf->getNumAtoms());
    std::vector<long long int> forceX(numAtoms);
    std::vector<long long int> forceY(numAtoms);
    std::vector<long long int> forceZ(numAtoms);
    bondedForces->getXYZ(forceX.data(), forceY.data(), forceZ.data());

    long long int netForceX = 0;
    long long int netForceY = 0;
    long long int netForceZ = 0;

    for (std::size_t atom = 0; atom < numAtoms; ++atom) {
      netForceX += forceX[atom];
      netForceY += forceY[atom];
      netForceZ += forceZ[atom];
    }

    CHECK(netForceX == 0);
    CHECK(netForceY == 0);
    CHECK(netForceZ == 0);
  }

  SECTION("Bonded gradient matches a Cartesian finite difference") {
    const auto bondedForces = forceManager->getBondedForcevalues();
    REQUIRE(bondedForces != nullptr);

    const std::size_t numAtoms = static_cast<std::size_t>(psf->getNumAtoms());
    std::vector<long long int> forceX(numAtoms);
    std::vector<long long int> forceY(numAtoms);
    std::vector<long long int> forceZ(numAtoms);
    bondedForces->getXYZ(forceX.data(), forceY.data(), forceZ.data());

    const std::vector<CrossTerm> &crossTerms = psf->getCrossTerms();
    REQUIRE(crossTerms.empty() == false);
    const int atom = crossTerms.front().iatom1;

    const std::vector<double3> originalCoordinates =
        coordinates->getCoordinatesDP();
    REQUIRE(atom >= 0);
    REQUIRE(static_cast<std::size_t>(atom) < originalCoordinates.size());

    const auto bondedEnergy = [](const std::map<std::string, double> &energy) {
      return energy.at("bond") + energy.at("ureyb") + energy.at("angle") +
             energy.at("dihe") + energy.at("imdihe") + energy.at("cmap");
    };

    constexpr double displacement = 1.0e-3;

    std::vector<double3> displacedCoordinates = originalCoordinates;
    displacedCoordinates[atom].x += displacement;
    context->setCoordinates(displacedCoordinates);
    context->calculateForces(false, true, false);
    const double energyPlus = bondedEnergy(forceManager->getEnergyComponents());

    displacedCoordinates = originalCoordinates;
    displacedCoordinates[atom].x -= displacement;
    context->setCoordinates(displacedCoordinates);
    context->calculateForces(false, true, false);
    const double energyMinus =
        bondedEnergy(forceManager->getEnergyComponents());

    const double numericalGradient =
        (energyPlus - energyMinus) / (2.0 * displacement);
    constexpr double inverseForceScale = 1.0 / static_cast<double>(1LL << 40);
    const double analyticalGradient =
        static_cast<double>(forceX[atom]) * inverseForceScale;

    CHECK(analyticalGradient == Approx(numericalGradient).margin(5.0e-3));
  }

  SECTION("CMAP gradients match OpenMM reference forces") {
    struct ForceReference {
      int atom;
      double x;
      double y;
      double z;
    };

    // OpenMM Reference platform CMAP physical forces (-dE/dr), in
    // kcal/mol/A, for every atom with a nonzero CMAP contribution.
    const ForceReference openmmForces[] = {
        {17, -2.651172559437, -0.393876366949, 1.326840767917},
        {19, 3.595409708583, 0.774928720518, -2.240989104374},
        {21, -2.642885008965, 0.885054471002, -0.034362253435},
        {34, 2.610664080130, -5.749670833753, 2.700995480862},
        {36, -2.455930868056, 6.711269572591, -3.387457404737},
        {38, -0.414271102374, -1.102528931522, -0.065865454590},
        {53, 2.315434845817, -1.031103103507, 1.861341118368},
        {55, 0.620638594527, -0.785162063014, 0.764451356420},
        {57, 0.886091837387, -0.104770450586, -1.667663445499},
        {73, -1.351668222232, -0.304474016231, 0.668515877899},
        {75, -0.635589407616, 2.092313938951, 0.110490474789},
        {77, -1.545421335456, -0.999377635805, 0.541692150480},
        {89, 1.085875780866, 0.053579520743, -0.901011755821},
        {91, 0.621459821166, -0.124100763667, 0.637210306363},
        {93, 0.553150092996, 0.196261523175, -0.570664399896},
        {111, -2.809623900870, 2.684976538758, 0.238675900844},
        {113, 1.546314188403, -7.387465945392, 1.767472427774},
        {115, 1.538799437923, 6.715556331650, -3.390471640137},
        {125, 0.701417215154, -0.168968664059, 1.492236333756},
        {127, -2.247241179620, -1.646357244767, -0.799582462556},
        {129, 0.839727766837, -0.185331524220, 1.620464632903},
        {144, -0.154816491954, 0.030786928263, -0.615995745292},
        {146, -0.895721446615, 0.086781581718, -0.773203522641},
        {148, 0.466590545646, -0.362253573176, 0.784654621252},
        {158, -0.154486478267, 1.795939797649, 0.530607770594},
        {160, 1.780198286736, -0.975577582687, -0.846779683131},
        {162, -1.648014502490, -1.502851862785, 0.677681168302},
        {165, 0.582603836023, 1.799863892378, -1.146280198730},
        {167, 0.272033595893, -1.043218012941, 0.977426138921},
        {169, -0.096074545403, -0.098375604062, -0.244695784396},
        {187, -2.747259033290, 1.813907192418, -0.139846231051},
        {189, 3.353802752805, -2.436869374150, 0.257708254167},
        {191, 0.046602208891, -0.179714539595, 0.989006333126},
        {201, -0.947156299896, 0.939025992565, -0.994819065683},
        {203, -0.524165266806, 1.010908452935, -1.408551531315},
        {205, 0.054326217926, -0.283994896889, 1.025845731862},
        {220, 1.349280168841, -2.119973199941, 0.593889768626},
        {222, -1.046421981583, 1.154937558408, 0.110721061579},
        {224, 0.197185119618, 0.277202646080, -0.183875087794},
        {234, -0.106221493254, 0.099034729247, 0.313562025835},
        {236, -0.083076899926, -0.288159212747, -0.922053340856},
        {238, 0.557118943829, 0.369866400923, -0.572676925122},
        {253, -0.435804839732, -0.215215308693, 0.892495779374},
        {255, 0.600192219949, -0.135953810413, -0.139528597439},
        {257, -0.869687807632, 0.032799436912, 0.111538114576},
        {268, -0.009351370268, 2.323241308758, 2.532022334551},
        {270, 2.711063844202, -4.444997394592, -4.833821483400},
        {272, -0.153038847732, 1.182696772755, 0.750303832039},
        {284, -4.309624776754, 1.755745833666, 2.989753599632},
        {286, 1.238718609958, -0.166997456094, -0.833085478530},
        {288, -0.188387611584, 0.580877525032, -0.393258344638},
        {299, 1.201413616120, -1.461960789318, -0.368142381714},
        {301, 1.352837508189, 2.728712156430, 2.002134333848},
        {305, -3.040239980222, -4.397469537967, -2.731346067704},
        {313, 3.955983380390, 4.797480343545, 1.897693228352},
        {315, -2.158080538078, -2.661595547704, -1.223185646532},
        {317, -0.595297822888, -0.116155964674, 0.632904787925},
        {324, -1.922003898913, 0.658012010310, 1.694392723457},
        {326, 3.162303913433, -0.668764723821, -3.257804909756},
        {328, -1.960301571780, 2.478200508210, 0.976985510467},
        {336, 2.297764291312, -0.460486273520, -0.729695839032},
        {338, -6.549091306119, -3.973370043449, 1.354192272701},
        {340, 8.312461154916, 2.044297831684, -1.060437549833},
        {350, 0.266913569132, -0.365130507080, 2.403818325651},
        {352, -3.411238803683, 0.207936726369, -1.538464470136},
        {354, 0.115965154426, -0.004108540794, -0.324514070234},
        {369, -0.193922436888, -0.582971289927, -0.687485347277},
        {371, 0.136792251722, -1.177130798969, 0.880500206241},
        {373, -0.971939613440, 2.734733571674, -0.301036223799},
        {384, 0.522490396144, -0.823888303303, 0.321029177642},
        {386, 0.611588542774, -1.816135139189, -0.376320736996},
        {388, -1.236233714746, 1.528078691283, 0.776070592057},
        {398, 0.988064496247, -0.174397471898, -0.491676082805},
        {400, -1.186931560712, -0.177464331875, -0.234035422093},
        {402, 1.370168035819, 0.002555844646, 1.046775227902},
        {414, -1.322167804102, 0.905706436407, -0.438153780212},
        {416, 1.469865172423, -1.820153705396, 1.190919542555},
        {418, -0.798869602467, 1.642522818523, -2.827409594880},
        {436, 0.521271842400, -0.437185412331, 1.871241663695},
        {438, 0.133240694824, -1.728443487335, -0.455556126165},
        {440, -1.011321239326, 2.258066741284, 0.059813138327},
        {446, 1.435316310355, -1.498062504844, -0.618968682130},
        {448, 0.399756645415, 0.478277371808, -0.329630667812},
        {450, -2.217997830287, 0.531705324124, 1.141047250790},
        {468, 5.722281467995, -0.456357510423, 2.326230078516},
        {470, -7.040169396707, 1.061153425582, -4.478229664925},
        {472, 2.583541368482, -1.826483514475, 4.591945767791},
        {487, -1.969345338740, 2.073835777879, 0.535825563397},
        {489, 6.148412829651, -4.746013342196, -6.214265930370},
        {491, -8.252504244853, 7.984282074770, 3.189903351498},
        {504, 8.363464571792, -6.404396114675, 1.971898412050},
        {506, -4.551113822983, 1.999587606082, -3.678653485570},
        {508, 0.340432430361, 1.008287746103, 2.579976497296},
        {516, 0.312502910535, -1.248401871408, -3.561612865497},
        {518, -4.856038531023, -1.266371402365, 1.982623369438},
        {520, 8.747195070983, 3.621403630892, 1.440668368151},
        {538, -9.651696818272, -2.860966526418, -3.004795328783},
        {540, 4.200846300203, -3.146841017205, 4.376347464933},
        {542, 3.020888649981, 7.829747431517, -5.074604604107},
        {553, -4.892004038316, -7.333873756745, 6.699394179125},
        {555, 3.338531671880, 3.946813447717, -4.387488439948},
        {557, -0.275103865196, -0.209006735922, 0.194231643175},
        {560, -0.482126372296, 0.765299316371, 0.036438201694},
        {562, -0.983194235034, -1.345001034723, -0.930309710692},
        {564, 2.195143245064, 0.902244942039, 2.362630912000},
        {579, -4.660528692030, 0.884050131044, 3.471417582570},
        {581, 5.714200022399, -2.803353489078, -14.279392056959},
        {585, -0.901891145074, 3.708444828676, 15.713294809043},
        {593, -0.758931445008, -1.837307446993, -7.396365584465},
        {595, -0.465758864396, 1.316465875159, 2.891655106676},
        {599, -0.836335134431, -1.180690132592, -2.159827215160},
        {607, 1.900550888942, 8.899933258509, -0.133753284187},
        {609, 1.216336539097, -10.761681774100, 1.175963505990},
        {611, -0.499433782240, 2.485297779266, -1.582837403041},
        {619, -0.466386934221, -1.004300405721, 1.581148380276},
        {621, -0.471228093646, 0.388231722460, 1.841367839714},
        {623, 0.742586113463, 0.116496591587, -4.719717716690},
        {636, -1.864794273880, -2.150708561621, 3.516920693722},
        {638, 0.954682947765, 3.052442213099, -1.877762374717},
        {640, -0.061817059641, -2.950121332378, -0.377114047333},
        {653, 0.350120657033, 2.022399021624, 2.136635034436},
        {655, -1.025848738715, 2.262383009429, -0.173262565005},
        {657, -0.232407355829, -3.966025246654, -0.561432314108},
        {677, 2.896454788164, 1.815711220041, -3.229719329329},
        {679, -2.429707491242, 0.732003177435, 3.187709449176},
        {681, 0.310729382806, -0.949171860429, -0.013055193227},
        {696, -0.230259616192, -0.591152351193, -0.782825785114},
        {698, 0.057729806027, -0.262340916800, 0.062808652072},
        {700, 0.272076436559, -0.046824429784, 0.304252781341},
        {715, -0.570248521503, 1.033332171752, 0.181232651825},
        {717, 0.277722952042, -0.197367292028, -0.107638253241},
        {719, 0.058890215383, -0.044478503984, 0.011557680992},
        {735, 0.041134434545, -0.271020058979, -0.219847556846},
        {737, -0.081749761400, 0.456110776006, 0.305632303596},
        {739, 0.205557953850, -0.457162878145, -0.189887171483},
        {745, -0.760151064217, 0.291527549395, -0.067884107045},
        {747, 0.208527142430, -1.510873508359, -0.405568755091},
        {749, 0.771346796148, 2.435623436400, 1.092851573885},
        {752, -1.684377504416, -4.617424284030, -3.458648662709},
        {754, 1.727656585391, 4.600269179679, 3.949610791303},
        {756, -0.874305361980, -0.009014494456, 1.015195928572},
        {774, 0.812334289382, -1.633438656251, -2.565595153767},
        {776, -0.351912172088, 0.722149992309, -0.218203399917},
        {778, -0.235613622205, -0.135031952359, 0.421573629975},
        {791, 0.009968871844, 0.316678632049, 0.074250249033},
        {793, 0.971913223717, -0.004898718641, 0.284129130887},
        {795, -1.293026442784, -0.991189502884, 0.432047071966},
        {810, 0.483128491523, -1.127052876121, -5.241938112139},
        {812, 0.475614546587, 3.438580446268, 7.603768586780},
        {814, -2.618525823757, -4.977918142612, -3.306756916281},
        {825, 2.326343261175, 5.535770215213, -0.124638211300},
        {827, 0.217307838619, -3.059824916212, 1.140797212065},
        {829, -0.994567957385, 1.330832624307, -1.275714718286},
        {837, 1.088468319481, -0.504981087408, 0.791675592957},
        {839, -0.116516014566, 0.474355698987, -1.110111266905},
        {841, -0.583281686320, -0.797903873189, 1.746674260163},
        {844, 2.616876323308, 4.058893381246, -5.154480209371},
        {846, -2.738119575447, -8.122584063311, 5.584951583757},
        {848, -0.009796245883, 8.309566592049, -0.432411269300},
        {868, -1.315093971501, -2.096838761243, -0.515749676214},
        {870, 3.394283935374, -0.761029446112, 0.041039404909},
        {872, -2.092559465262, -1.361512031559, -0.257229112748},
        {882, -2.059068980228, -3.057052207778, -1.469902952278},
        {884, 2.939280403477, 4.793738818017, 2.641553436537},
        {886, 0.021417246694, -2.083631688423, -3.010964157785},
        {901, 0.066730841818, 0.084122444288, 4.053955774882},
        {903, 0.003417361292, -0.320880555985, -3.134541478381},
        {905, 0.200374300421, 1.143445152356, 1.255290892971},
        {912, -1.227100873525, -4.036603664415, -0.778492340384},
        {914, 1.466173119959, 4.884276538210, 0.805612465917},
        {916, -2.176865261813, -1.523888523974, -0.924714156328},
        {924, 2.839605417163, 0.568880542315, 2.761037729282},
        {926, -0.381725979535, 1.160411166014, -4.089007182952},
        {928, -3.637298507063, -2.854937941016, 3.298494005635},
        {945, 0.532218175282, -0.016572923924, -0.074017755544},
        {947, 5.151060278990, 0.900166864022, -4.814977935624},
        {949, -5.093362528225, 3.825667971929, 7.178550624758},
        {959, 1.336622790384, -4.107978229199, -5.468858752196},
        {961, 0.254116287202, 1.037109887608, 1.671322255494},
        {963, 0.005604017918, -0.031968448128, 0.114770108755},
        {978, -2.488884943326, -0.929657252912, 0.838159025195},
        {980, 5.770450437542, 0.574234021509, -1.613475148198},
        {982, -0.404463043351, -1.031906483860, 0.560726867980},
        {995, -8.554978277069, 4.039807243299, -0.141963740958},
        {997, 8.025897743390, -3.058709765692, 1.390283776584},
        {999, -3.822657157430, -0.217193134705, -3.212746972721},
        {1017, 0.844092456588, 1.353019232836, -1.847376147141},
        {1019, -0.063692868403, -1.499366358090, 5.954784471408},
        {1021, 0.016671133875, 0.754637897669, -2.123755848588},
        {1032, 1.682261587280, -0.154350388283, -0.342314199730},
        {1034, -3.409203721694, 0.328751336474, 0.493534073056},
        {1036, 4.769775874610, -2.250062851508, 0.333325977226},
        {1043, -2.871847977734, 3.591394449654, -1.693818803553},
        {1045, 0.288867505269, -1.764231515176, 2.398256340197},
        {1047, 1.769558686844, -0.350049056549, -1.684113084305},
        {1057, -1.504986969586, 0.043921521865, 0.523946910474},
        {1059, 0.302213658997, 0.643028157385, 0.101140677866},
        {1061, -0.291199785052, -0.074002602213, 0.162760405002},
        {1076, -0.199529685886, 0.236041712867, -0.302108420639},
        {1078, 0.457356182399, -0.301876790467, 0.227965104433},
        {1080, 0.019030496800, 0.062826815902, -0.226009155090},
        {1093, 0.220510853363, -0.972562412472, -1.109103463805},
        {1095, -0.290637544796, 1.969170781390, 1.621141424182},
        {1097, -0.697353080180, -1.882336273274, 0.601096753636},
        {1112, 0.794565343783, 0.821767279695, -0.956140419648},
        {1114, -0.011777706380, 1.078272310918, -0.541968357690},
        {1116, -0.196061285871, -1.253644501633, 0.367585397994},
        {1128, -0.300284128206, -0.239461914345, 1.410459622403},
        {1130, 0.224774443293, -0.293117310909, -1.427676764655},
        {1132, -0.052250298678, 0.812637395046, 0.259709421130},
        {1147, -0.292350801512, -0.532279493740, 0.291644661047},
        {1149, 0.632214448027, 2.490034330151, 0.053365891189},
        {1151, -0.588942360654, -0.621870538195, -0.301885074762},
        {1171, 0.881224403836, -2.631028322834, 0.079564149421},
        {1173, -0.235760107703, 0.645245250486, -1.364456932938},
        {1175, 0.390386152745, 1.004040428589, 1.174611750896},
        {1190, 1.769469079110, 1.480005835403, 0.183671938350},
        {1192, -5.378959866202, -4.026162591902, -0.302029854171},
        {1194, 0.974688888136, 1.435230273269, 0.056374269061},
        {1214, 1.372562928670, 0.268614937744, -0.776039866797},
        {1216, 0.940829822399, -0.247962318339, 1.143596403065},
        {1218, -1.784534586339, -0.346849318033, 0.150433324814},
        {1221, 1.513263758437, 1.058525239789, -0.668862504464},
        {1223, -0.038966293540, -0.009728857654, 0.246928891361},
    };

    const BondedParamsAndLists bondedData =
        parameters->getBondedParamsAndLists(psf);
    const auto bondedStream = forceManager->getBondedStream();
    REQUIRE(bondedStream != nullptr);

    CudaEnergyVirial energyVirial;
    CudaBondedForce<long long int, float> cmapForce(
        energyVirial, nullptr, nullptr, nullptr, nullptr, nullptr, "cmap");
    cmapForce.setup_list(bondedData.listsSize, bondedData.listVal,
                         *bondedStream);
    cmapForce.setup_coef(bondedData.paramsSize, bondedData.paramsVal);

    Force<long long int> cmapForceValues(psf->getNumAtoms());
    cmapForceValues.clear(*bondedStream);

    const float4 *xyzq =
        context->getCoordinatesChargesSP().getDeviceArray().data();
    cmapForce.calc_force(
        xyzq, 50.0f, 50.0f, 50.0f, false, false, cmapForceValues.stride(),
        cmapForceValues.xyz(), false, false, false, false, false, true,
        *bondedStream);
    cudaCheck(cudaStreamSynchronize(*bondedStream));

    const std::size_t numAtoms = static_cast<std::size_t>(psf->getNumAtoms());
    std::vector<long long int> forceX(numAtoms);
    std::vector<long long int> forceY(numAtoms);
    std::vector<long long int> forceZ(numAtoms);
    cmapForceValues.getXYZ(forceX.data(), forceY.data(), forceZ.data());

    constexpr double inverseForceScale = 1.0 / static_cast<double>(1LL << 40);
    constexpr double tolerance = 5.0e-3;

    for (const ForceReference &reference : openmmForces) {
      REQUIRE(reference.atom >= 0);
      REQUIRE(static_cast<std::size_t>(reference.atom) < numAtoms);

      const double gradientX =
          static_cast<double>(forceX[reference.atom]) * inverseForceScale;
      const double gradientY =
          static_cast<double>(forceY[reference.atom]) * inverseForceScale;
      const double gradientZ =
          static_cast<double>(forceZ[reference.atom]) * inverseForceScale;

      CHECK(gradientX == Approx(-reference.x).margin(tolerance));
      CHECK(gradientY == Approx(-reference.y).margin(tolerance));
      CHECK(gradientZ == Approx(-reference.z).margin(tolerance));
    }
  }

  SECTION("CMAP virial is invariant to a shifted atom representation") {
    const auto bondedStream = forceManager->getBondedStream();
    REQUIRE(bondedStream != nullptr);

    const auto calculateCmapVirial =
        [&](const BondedParamsAndLists &bondedData) {
          CudaEnergyVirial energyVirial;
          CudaBondedForce<long long int, float> cmapForce(
              energyVirial, nullptr, nullptr, nullptr, nullptr, nullptr,
              "cmap");
          cmapForce.setup_list(bondedData.listsSize, bondedData.listVal,
                               *bondedStream);
          cmapForce.setup_coef(bondedData.paramsSize, bondedData.paramsVal);

          Force<long long int> cmapForceValues(psf->getNumAtoms());
          cmapForceValues.clear(*bondedStream);
          energyVirial.clear(*bondedStream);

          const float4 *xyzq =
              context->getCoordinatesChargesSP().getDeviceArray().data();
          cmapForce.calc_force(
              xyzq, 50.0f, 50.0f, 50.0f, false, true,
              cmapForceValues.stride(), cmapForceValues.xyz(), false, false,
              false, false, false, true, *bondedStream);
          cudaCheck(cudaStreamSynchronize(*bondedStream));

          cmapForceValues.convert<double>(*bondedStream);
          cudaCheck(cudaStreamSynchronize(*bondedStream));

          energyVirial.calcVirial(
              psf->getNumAtoms(), xyzq, 50.0, 50.0, 50.0,
              cmapForceValues.stride(),
              reinterpret_cast<double *>(cmapForceValues.xyz()),
              *bondedStream);
          energyVirial.copyToHost(*bondedStream);
          cudaCheck(cudaStreamSynchronize(*bondedStream));

          std::array<double, 9> virial{};
          energyVirial.getVirial(virial.data());
          return virial;
        };

    BondedParamsAndLists bondedData =
        parameters->getBondedParamsAndLists(psf);
    const std::array<double, 9> referenceVirial = calculateCmapVirial(bondedData);

    std::size_t cmapOffset = 0;
    for (int term = 0; term < 5; ++term)
      cmapOffset += static_cast<std::size_t>(bondedData.listsSize[term]);

    REQUIRE(cmapOffset < bondedData.listVal.size());
    REQUIRE(bondedData.listVal[cmapOffset].size() == 15);

    const int atom = psf->getCrossTerms().front().iatom1;
    REQUIRE(bondedData.listVal[cmapOffset][0] == atom);

    // Shift code 12 adds -boxX to i1, reconstructing its original image.
    bondedData.listVal[cmapOffset][9] = 12;

    std::vector<double3> imagedCoordinates = coordinates->getCoordinatesDP();
    imagedCoordinates[atom].x += 50.0;
    context->setCoordinates(imagedCoordinates);

    const std::array<double, 9> imagedVirial = calculateCmapVirial(bondedData);

    for (std::size_t component = 0; component < referenceVirial.size();
         ++component) {
      CHECK(imagedVirial[component] ==
            Approx(referenceVirial[component]).margin(1.0e-4));
    }
  }
}
