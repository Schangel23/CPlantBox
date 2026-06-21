// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#include "tropism.h"

#include "soil.h"
#include "sdf.h"
#include "Root.h"
#include "Leaf.h"
#include "leafparameter.h"
#include <stdlib.h>
#include <cstdlib>

namespace CPlantBox {

/**
 * Copies this tropism
 */
std::shared_ptr<Tropism> Tropism::copy(std::shared_ptr<Organism> plant)
{
    auto nt = std::make_shared<Tropism>(*this); // default copy constructor
    nt->plant  =  std::weak_ptr<Organism>(); // necessary?
    nt->plant = plant;
    // std::cout << "Base class - Copy tropism: from " << this->plant.lock()->plantId << " to " << nt->plant.lock()->plantId << "\n";
    return nt;
}

/**
 * Applies angles a and b and goes dx [cm] into the new direction and returns the new position
 *
 * @param pos          root tip position
 * @param old          rotation matrix, heading is old(:,1)
 * @param a            angle alpha, describes the angular rotation of the axis old
 * @param b            angle beta, describes the radial rotation of the axis old
 * @param dx           distance to look ahead
 *
 * \return             position
 */
Vector3d Tropism::getPosition(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx)
{
    return pos.plus((old.times(Vector3d::rotAB(a,b))).times(dx));
}

/**
 * Dices N times picking angles alpha and beta, takes the optimal direction according to the objective function
 *
 * @param pos          root tip position
 * @param old          rotation matrix, heading is old(:,1)
 * @param dx           distance to look ahead (e.g in case of hydrotropism)
 * @param root         points to the root that called getHeading(...), just in case something else is needed (i.e. iheading for exotropism)
 * @param nodeIdx    index of the node on which to apply the tropism
 *
 * \return             angle alpha and beta
 */
Vector2d Tropism::getUCHeading(const Vector3d& pos, const Matrix3d& old, double dx, const std::shared_ptr<Organ> o, int nodeIdx)
{
    double a = sigma*randn(nodeIdx)*sqrt(dx); // <-- causes a segmentation fault if plant is expired
    double b = rand(nodeIdx)*2*M_PI;
    double v;
    double n_=n*sqrt(dx);
    if (n_>0) {
        double dn = n_-floor(n_);
        if (rand(nodeIdx)<dn) {
            n_ = ceil(n_);
        } else {
            n_ = floor(n_);
        }
        double bestA = a;
        double bestB = b;
        double bestV = this->tropismObjective(pos,old,a,b,dx,o);
        for (int i=0; i<n_; i++) {
            b = rand(nodeIdx)*2*M_PI;
            a = sigma*randn(nodeIdx)*sqrt(dx);
            v = this->tropismObjective(pos,old,a,b,dx,o);
            if (v<bestV) {
                bestV=v;
                bestA=a;
                bestB=b;
            }
        }
        a = bestA;
        b = bestB;
    }
    return Vector2d(a,b);
}

/**
 * Inside the geometric domain baseTropism::getHeading() is returned.
 * In case geometric boundaries are hit, the rotations are modified, so that growth stays inside the domain.
 *
 * @param pos        root tip position
 * @param old        rotation matrix, heading is old(:,1)
 * @param dx         distance to look ahead (e.g in case of hydrotropism)
 * @param root       points to the root that called getHeading(), just in case something else is needed (i.e. iheading for exotropism)
 * @param nodeIdx    index of the node on which to apply the tropism
 *
 * \return           the rotations alpha and beta
 */
Vector2d Tropism::getHeading(const Vector3d& pos, const Matrix3d& old, double dx, const std::shared_ptr<Organ> o, int nodeIdx)
{
    if (nodeIdx > 0){
        gen =  std::mt19937(plant.lock()->getSeedVal() + nodeIdx + o->getId());
    }
    Vector2d h = this->getUCHeading(pos, old, dx, o, nodeIdx);
    double a = h.x;
    double b = h.y;
    if (!geometry.expired()) {
        double d = geometry.lock()->getDist(this->getPosition(pos,old,a,b,dx));
        double dmin = d;

        double bestA = a;
        double bestB = b;
        int i=0; // counts change in alpha
        int j=0;    // counts change in beta

        while ((d>0)&&(o->organType()== Organism::ot_root))  { // not valid
            i++;
            j=0;
            while ((d>0) && j<betaN) { // change beta

                b = 2*M_PI*rand(nodeIdx); // dice
                d = geometry.lock()->getDist(this->getPosition(pos,old,a,b,dx));
                if (d<dmin) {
                    dmin = d;
                    bestA = a;
                    bestB = b;
                }
                j++;
            }

            if ((d>0)&&(o->organType()== Organism::ot_root))  {
                a = a + M_PI/2./double(alphaN);
            }

            if (i>alphaN) {
                //std::cout << "Could not respect geometry boundaries \n";
                a = bestA;
                b = bestB;
                break;
            }

        }
    }
    return Vector2d(a,b);
}



/**
 * getHeading() minimizes this function, @see TropismFunction::tropismObjective
 */
double Exotropism::tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx, const std::shared_ptr<Organ> o)
{
    Vector3d iheading =o->getiHeading0();
    double s = iheading.times(old.times(Vector3d::rotAB(a,b)));
    s*=(1./iheading.length()); // iheading should be normed anyway?
    s*=(1./old.column(0).length());
    return acos(s)/M_PI; // 0..1
}


/**
 * DistalGravitropism: gravitropic pull that grades from weak (stiff, keeps the
 * insertion heading) at the leaf base to strong (droops down) at the tip, so the
 * blade arches over distally. @see DistalGravitropism in tropism.h. Defined here
 * (not inline) because it needs LeafSpecificParameter::getK() for the arc-length
 * fraction. Falls back to plain gravitropism for non-leaf organs.
 */
double DistalGravitropism::tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx, const std::shared_ptr<Organ> o)
{
    Vector3d nh = old.times(Vector3d::rotAB(a,b));   // candidate new heading
    double gravi = 0.5*(nh.z + 1.);                  // [0,1], 0 when pointing straight down

    // keep-insertion-heading term (stiff proximal tissue), as in Exotropism
    Vector3d ih = o->getiHeading0();
    double ihl = ih.length(); if (ihl < 1e-9) ihl = 1.;
    double ocl = old.column(0).length(); if (ocl < 1e-9) ocl = 1.;
    double s = ih.times(nh) / (ihl * ocl);
    if (s > 1.) s = 1.; if (s < -1.) s = -1.;
    double exo = std::acos(s)/M_PI;                  // [0,1], 0 when aligned with insertion heading

    // distal grading: frac = arc-length position of the node laid this growth step
    double frac = 0.5;
    auto lp = std::dynamic_pointer_cast<const LeafSpecificParameter>(o->param());
    if (lp) {
        double Lmax = lp->getK(); if (Lmax < 1e-9) Lmax = 1e-9;
        frac = o->getLength(true) / Lmax;
        if (frac < 0.) frac = 0.; if (frac > 1.) frac = 1.;
    }
    double g = std::pow(frac, std::max(ageSwitch, 1e-6));   // ageSwitch slot carries distalExp

    return (1.0 - g)*exo + g*gravi;                  // base: keep heading; tip: go down
}


/**
 * PrescribedMidribTropism: returns the exact (alpha,beta) that steers the next
 * growth step along the leaf's surface_cps v=midrib tangent at the current
 * arc-length fraction, mapped to world space through the same insertion frame as
 * Leaf::updateNodesFromSurfaceCPs. Deterministic (dice bypassed). @see
 * PrescribedMidribTropism in tropism.h. Keep-heading fallback when unavailable.
 */
Vector2d PrescribedMidribTropism::getUCHeading(const Vector3d& pos, const Matrix3d& old, double dx,
	const std::shared_ptr<Organ> o, int nodeIdx)
{
	if (!o) return Vector2d(0., 0.);
	auto lrp = std::dynamic_pointer_cast<LeafRandomParameter>(o->getOrganRandomParameter());
	if (!lrp) return Vector2d(0., 0.);
	const int n_u = lrp->surface_n_u;
	const int n_v = lrp->surface_n_v;
	const int v_mid = n_v / 2;
	if (n_u < 2 || n_v < 1) return Vector2d(0., 0.);

	// Use the SAME shape source as Leaf::updateNodesFromSurfaceCPs — the
	// maturity-blended effective grid, which is the FP4D ParametricLeafShape
	// when a shape_distribution is loaded (NOT the raw pre-distribution LRP
	// grid). Falls back to the raw grid if o is not a Leaf / effective empty.
	std::vector<Vector3d> eff;
	auto leaf = std::dynamic_pointer_cast<Leaf>(o);
	if (leaf) eff = (useMatureShape && lrp->use_tropism_midrib) ? leaf->getEffectiveSurfaceCPs(1.) : leaf->getEffectiveSurfaceCPs();
	if ((int)eff.size() != n_u * n_v) eff = lrp->surface_cps;
	if ((int)eff.size() < n_u * n_v) return Vector2d(0., 0.);

	// midrib polyline (leaf-local) + arc-length
	std::vector<Vector3d> mid; mid.reserve(n_u);
	for (int iu = 0; iu < n_u; ++iu) mid.push_back(eff[iu * n_v + v_mid]);
	std::vector<double> cum(n_u, 0.);
	for (int iu = 1; iu < n_u; ++iu) {
		Vector3d d = mid[iu].minus(mid[iu - 1]);
		cum[iu] = cum[iu - 1] + std::sqrt(d.times(d));
	}
	const double total = cum.back();
	if (total < 1e-9) return Vector2d(0., 0.);

	// arc-length fraction of the growing tip → local midrib tangent there
	const double Lmax = std::max(lrp->lmax, 1e-9);
	double frac = o->getLength(true) / Lmax;
	if (frac < 0.) frac = 0.; if (frac > 1.) frac = 1.;
	const double target_s = frac * total;
	// Smooth, continuously-varying tangent at target_s. A piecewise-constant
	// per-CP-segment tangent (mid[k+1]-mid[k]) is discontinuous at CP-segment
	// crossovers: on a non-arc-uniform CP polyline it dumps the whole
	// inter-segment turn into one growth step (kinks / spurious kappa spikes,
	// e.g. the rank-10 apex pendant turn). Instead take a centered difference of
	// the arc-interpolated midrib over a window of ~one mean CP-segment, so the
	// curvature is distributed smoothly along the arc (faithful kappa(s)).
	auto midAt = [&](double s) -> Vector3d {
		if (s <= 0.) return mid[0];
		if (s >= total) return mid[n_u - 1];
		int j = 0;
		while (j < n_u - 2 && cum[j + 1] < s) ++j;
		const double seg = cum[j + 1] - cum[j];
		const double w = (seg > 1e-12) ? (s - cum[j]) / seg : 0.;
		return mid[j].times(1. - w).plus(mid[j + 1].times(w));
	};
	const double hw = 0.5 * total / (double)(n_u - 1);   // half-window ~ 1/2 mean CP-segment
	Vector3d tan_l = midAt(target_s + hw).minus(midAt(target_s - hw));
	double tll = std::sqrt(tan_l.times(tan_l));
	if (tll < 1e-12) {                                    // degenerate window -> fall back to chord
		tan_l = mid[n_u - 1].minus(mid[0]);
		tll = std::sqrt(tan_l.times(tan_l));
	}
	if (tll < 1e-12) return Vector2d(0., 0.);
	tan_l = tan_l.times(1. / tll);

	// insertion frame (world), identical convention to Leaf::updateNodesFromSurfaceCPs
	Vector3d tangent = o->getiHeading0();
	double tl = std::sqrt(tangent.times(tangent));
	if (tl < 1e-9) return Vector2d(0., 0.);
	tangent = tangent.times(1. / tl);
	Vector3d up(0., 0., 1.);
	Vector3d x_local = tangent.cross(up);
	double xl = std::sqrt(x_local.times(x_local));
	if (xl < 1e-6) {
		Vector3d alt = (std::abs(tangent.x) < 0.9) ? Vector3d(1., 0., 0.) : Vector3d(0., 1., 0.);
		x_local = tangent.cross(alt);
		xl = std::sqrt(x_local.times(x_local));
	}
	if (xl < 1e-12) return Vector2d(0., 0.);
	x_local = x_local.times(1. / xl);
	Vector3d y_local = tangent.cross(x_local);
	double yl = std::sqrt(y_local.times(y_local));
	if (yl < 1e-12) return Vector2d(0., 0.);
	y_local = y_local.times(1. / yl);

	// target world direction = R * tan_l, R = [x_local | y_local | tangent]
	Vector3d tw(
		x_local.x * tan_l.x + y_local.x * tan_l.y + tangent.x * tan_l.z,
		x_local.y * tan_l.x + y_local.y * tan_l.y + tangent.y * tan_l.z,
		x_local.z * tan_l.x + y_local.z * tan_l.y + tangent.z * tan_l.z);
	double twl = std::sqrt(tw.times(tw));
	if (twl < 1e-12) return Vector2d(0., 0.);
	tw = tw.times(1. / twl);

	// express target in the current ons frame: local = old^T * tw (orthonormal columns)
	double lx = old.column(0).times(tw);
	double ly = old.column(1).times(tw);
	double lz = old.column(2).times(tw);
	if (lx > 1.) lx = 1.; if (lx < -1.) lx = -1.;
	// rotAB(a,b) = (cos a, sin a cos b, sin a sin b); invert to hit (lx,ly,lz)
	const double a = std::acos(lx);
	const double b = std::atan2(lz, ly);
	return Vector2d(a, b);
}


/**
 * getHeading() minimizes this function, @see TropismFunction::tropismObjectiveMatrix3d
 */
double Hydrotropism::tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx, const std::shared_ptr<Organ> o)
{
    assert(!soil.expired());
    Vector3d newpos = this->getPosition(pos,old,a,b,dx);
    double v = soil.lock()->getValue(newpos,o);
    // std::cout << "\n" << newpos.getString() << ", = "<< v;
    return -v; ///< (-1) because we want to maximize the soil property
}



/**
 * Constructs a combined tropism with two tropsims,
 * the new objective funciton is (t1->tropismObjective()*w1)+(t2->tropismObjective()*w2)
 *
 * @param n             number of tries
 * @param sigma         standard deviation of angular change [1/cm]
 * @param t1            first tropism
 * @param w1            first weight
 * @param t2            first tropism
 * @param w2            first weight
 */
CombinedTropism::CombinedTropism(std::shared_ptr<Organism> plant, double n, double sigma, std::shared_ptr<Tropism> t1, double w1, std::shared_ptr<Tropism> t2, double w2)
    :Tropism(plant,n,sigma)
{
    tropisms.push_back(t1);
    tropisms.push_back(t2);
    weights.push_back(w1);
    weights.push_back(w2);
}

/**
 * getHeading() minimizes this function, @see TropismFunction::tropismObjective
 */
double CombinedTropism::tropismObjective(const Vector3d& pos, const Matrix3d& old, double a, double b, double dx, const std::shared_ptr<Organ> o)
{
    //std::cout << "CombinedTropsim::tropismObjective\n";
    double v = tropisms[0]->tropismObjective(pos,old,a,b,dx,o)*weights[0];
    for (size_t i = 1; i< tropisms.size(); i++) {
        v += tropisms[i]->tropismObjective(pos,old,a,b,dx,o)*weights[i];
    }
    return v;
}

} // end namespace CPlantBox
